"""
tests/integration/test_pipeline_smoke.py
=========================================
Integration smoke test: runs the full pipeline end-to-end
using fully synthetic data so no external API calls are needed.

This verifies:
  1. Feature pipeline produces valid FeatureFrame
  2. HMM, GARCH, SARIMAX, Bayesian all fit without error
  3. All models produce ForecastRecord
  4. Walk-forward backtest completes
  5. Experiment tracker records run
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.schemas import CoffeeVariety, DataFrequency, FeatureFrame


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _synthetic_arabica_raw(n_days: int = 500, seed: int = 42) -> dict:
    """
    Generate a minimal but realistic set of raw DataFrames
    simulating what ingestion would produce.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)

    # Arabica futures (cents/lb)
    log_ret = rng.normal(0.0003, 0.015, n_days)
    close = 185.0 * np.exp(np.cumsum(log_ret))
    arabica_df = pd.DataFrame({
        "open":   close * rng.uniform(0.998, 1.002, n_days),
        "high":   close * rng.uniform(1.001, 1.010, n_days),
        "low":    close * rng.uniform(0.990, 0.999, n_days),
        "close":  close,
        "volume": rng.integers(10000, 80000, n_days).astype(float),
        "variety": "arabica",
        "exchange": "ICE_NY",
    }, index=idx)

    # Robusta futures (USD/tonne)
    rob_ret = rng.normal(0.0001, 0.012, n_days)
    rob_close = 2100.0 * np.exp(np.cumsum(rob_ret))
    robusta_df = pd.DataFrame({
        "open":  rob_close * rng.uniform(0.998, 1.002, n_days),
        "high":  rob_close * rng.uniform(1.001, 1.008, n_days),
        "low":   rob_close * rng.uniform(0.992, 0.999, n_days),
        "close": rob_close,
        "volume": rng.integers(5000, 30000, n_days).astype(float),
    }, index=idx)

    # USD/BRL FX
    brl_base = 5.2
    brl_rate = brl_base + np.cumsum(rng.normal(0, 0.02, n_days))
    fx_usdbrl = pd.DataFrame({"rate": brl_rate, "pair": "USDBRL"}, index=idx)

    # EUR/USD
    eur_rate = 1.08 + np.cumsum(rng.normal(0, 0.003, n_days))
    fx_eurusd = pd.DataFrame({"rate": eur_rate, "pair": "EURUSD"}, index=idx)

    # Monthly ENSO ONI
    enso_idx = pd.date_range("2021-01-01", periods=n_days // 22, freq="MS")
    oni_vals = rng.normal(0.1, 0.6, len(enso_idx))
    enso_df = pd.DataFrame({
        "oni": oni_vals,
        "enso_phase": ["neutral"] * len(enso_idx),
    }, index=enso_idx)

    # Weekly COT
    cot_idx = pd.bdate_range("2021-01-04", periods=n_days // 5, freq="W-TUE")
    cot_df = pd.DataFrame({
        "noncommercial_long":  rng.integers(30000, 70000, len(cot_idx)).astype(float),
        "noncommercial_short": rng.integers(20000, 50000, len(cot_idx)).astype(float),
        "commercial_long":     rng.integers(40000, 80000, len(cot_idx)).astype(float),
        "commercial_short":    rng.integers(50000, 90000, len(cot_idx)).astype(float),
        "nonreportable_long":  rng.integers(5000, 15000, len(cot_idx)).astype(float),
        "nonreportable_short": rng.integers(5000, 15000, len(cot_idx)).astype(float),
        "open_interest":       rng.integers(100000, 200000, len(cot_idx)).astype(float),
        "variety": "arabica",
    }, index=cot_idx)

    return {
        "arabica_futures": arabica_df,
        "robusta_futures": robusta_df,
        "fx_usdbrl": fx_usdbrl,
        "fx_eurusd": fx_eurusd,
        "enso": enso_df,
        "cot_arabica": cot_df,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_data():
    return _synthetic_arabica_raw(n_days=600)


@pytest.fixture(scope="module")
def feature_frame(raw_data):
    from features.pipeline import FeaturePipeline

    fp = FeaturePipeline()
    return fp.run(
        raw_inputs=raw_data,
        variety=CoffeeVariety.ARABICA,
        frequency=DataFrequency.DAILY,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_frame_not_empty(self, feature_frame):
        assert not feature_frame.df.empty

    def test_frame_has_datetime_index(self, feature_frame):
        assert isinstance(feature_frame.df.index, pd.DatetimeIndex)

    def test_frame_has_target_column(self, feature_frame):
        target_cols = [c for c in feature_frame.df.columns if "target" in c]
        assert len(target_cols) > 0, "No target columns found"

    def test_price_features_present(self, feature_frame):
        cols = feature_frame.df.columns
        assert any("log_return" in c for c in cols)
        assert any("realised_vol" in c for c in cols)

    def test_climate_features_present(self, feature_frame):
        cols = feature_frame.df.columns
        assert "oni" in cols

    def test_no_all_nan_columns(self, feature_frame):
        all_nan_cols = feature_frame.df.columns[feature_frame.df.isna().all()].tolist()
        assert all_nan_cols == [], f"All-NaN columns: {all_nan_cols}"

    def test_feature_store_roundtrip(self, feature_frame):
        from features.store import ParquetFeatureStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetFeatureStore(tmpdir)
            store.save(feature_frame, "test_frame")
            assert store.exists("test_frame")
            loaded = store.load("test_frame")
            pd.testing.assert_frame_equal(
                feature_frame.df.select_dtypes(include=[float]),
                loaded.df.select_dtypes(include=[float]),
                check_freq=False,   # parquet roundtrip drops DatetimeIndex freq
            )


class TestHMMModel:
    def test_fits_and_predicts(self, feature_frame):
        pytest.importorskip("hmmlearn")
        from models.hmm_model import HMMRegimeDetector

        detector = HMMRegimeDetector(n_regimes=3, variety=CoffeeVariety.ARABICA)
        detector.fit(feature_frame)
        snapshots = detector.predict(feature_frame)
        assert len(snapshots) > 0
        # Each snapshot has a valid regime
        from contracts.schemas import MarketRegime
        for s in snapshots:
            assert s.regime in MarketRegime
            assert 0.0 <= s.probability <= 1.0

    def test_latest_regime(self, feature_frame):
        pytest.importorskip("hmmlearn")
        from models.hmm_model import HMMRegimeDetector

        detector = HMMRegimeDetector(n_regimes=2, variety=CoffeeVariety.ARABICA)
        detector.fit(feature_frame)
        latest = detector.predict_latest(feature_frame)
        assert latest.timestamp is not None
        assert 0.0 <= latest.probability <= 1.0


class TestGARCHModel:
    def test_fits_and_forecasts(self, feature_frame):
        pytest.importorskip("arch")
        from models.garch_model import GARCHVolatilityModel

        returns = feature_frame.df["log_return_1d"].dropna()
        model = GARCHVolatilityModel(variety=CoffeeVariety.ARABICA)
        model.fit(returns)
        fc = model.forecast(returns, horizon_days=10)
        assert fc.forecast_annualized_vol > 0
        assert fc.vol_regime in ("low", "medium", "high", "extreme")


class TestSARIMAXModel:
    def test_fits_and_predicts(self, feature_frame):
        pytest.importorskip("statsmodels")
        from models.sarimax_model import SARIMAXForecaster

        model = SARIMAXForecaster(order=(1, 0, 0), variety=CoffeeVariety.ARABICA)
        model.fit(feature_frame, target_col="target_log_return_5d")
        fc = model.predict(feature_frame, horizon_days=5)
        assert -0.5 < fc.mean_return < 0.5
        assert 0.0 <= fc.prob_up <= 1.0
        assert fc.q10 < fc.q90


class TestBayesianModel:
    def test_fits_and_predicts(self, feature_frame):
        pytest.importorskip("sklearn")
        from models.bayesian_model import BayesianForecaster

        model = BayesianForecaster(variety=CoffeeVariety.ARABICA)
        model.fit(feature_frame, target_col="target_log_return_5d")
        fc = model.predict(feature_frame, horizon_days=5)
        assert 0.0 <= fc.prob_up <= 1.0
        assert fc.q10 <= fc.mean_return <= fc.q90

    def test_feature_importance_non_empty(self, feature_frame):
        pytest.importorskip("sklearn")
        from models.bayesian_model import BayesianForecaster

        model = BayesianForecaster(variety=CoffeeVariety.ARABICA)
        model.fit(feature_frame, target_col="target_log_return_5d")
        imp = model.feature_importance()
        assert len(imp) > 0
        assert imp.max() > 0


class TestWalkForward:
    def test_backtest_completes(self, feature_frame):
        pytest.importorskip("statsmodels")
        from backtesting.walk_forward import WalkForwardBacktester
        from models.sarimax_model import SARIMAXForecaster

        backtester = WalkForwardBacktester(
            train_window=200,
            test_window=30,
            step=30,
        )
        model = SARIMAXForecaster(order=(1, 0, 0), variety=CoffeeVariety.ARABICA)
        result = backtester.run(
            model=model,
            frame=feature_frame,
            target_col="target_log_return_5d",
            n_folds=3,
        )
        assert result.n_folds == 3
        assert 0.0 <= result.directional_accuracy <= 1.0


class TestExperimentTracker:
    def test_full_lifecycle(self):
        from experiment.tracker import FileExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = FileExperimentTracker(tmpdir)
            run_id = tracker.start_run("test_run", tags={"variety": "arabica"})
            tracker.log_params(run_id, {"order": (1, 0, 1), "n_regimes": 3})
            tracker.log_metrics(run_id, {"directional_accuracy": 0.57, "brier": 0.23})
            tracker.end_run(run_id)

            runs = tracker.list_runs()
            assert len(runs) == 1
            assert runs[0]["status"] == "finished"

            val = tracker.get_latest_metric(run_id, "directional_accuracy")
            assert val == pytest.approx(0.57)
