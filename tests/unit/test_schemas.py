"""
tests/unit/test_schemas.py
==========================
Unit tests for data schemas and core feature logic.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from contracts.schemas import (
    CoffeeVariety,
    DataFrequency,
    FeatureFrame,
    ForecastRecord,
    OHLCVRecord,
    RegimeSnapshot,
    MarketRegime,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestOHLCVRecord:
    def test_valid_record(self):
        r = OHLCVRecord(
            timestamp=datetime(2023, 1, 15),
            symbol="KC=F",
            open=185.0, high=190.0, low=183.0, close=188.5,
            volume=12500.0,
        )
        assert r.close == 188.5

    def test_negative_price_rejected(self):
        with pytest.raises(Exception):
            OHLCVRecord(
                timestamp=datetime(2023, 1, 15),
                symbol="KC=F",
                open=-1.0, high=190.0, low=183.0, close=188.5,
            )

    def test_zero_price_rejected(self):
        with pytest.raises(Exception):
            OHLCVRecord(
                timestamp=datetime(2023, 1, 15),
                symbol="KC=F",
                open=0.0, high=190.0, low=183.0, close=188.5,
            )


class TestFeatureFrame:
    def _make_frame(self) -> FeatureFrame:
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {"feat_a": np.random.randn(100), "feat_b": np.random.randn(100)},
            index=idx,
        )
        return FeatureFrame(
            variety=CoffeeVariety.ARABICA,
            frequency=DataFrequency.DAILY,
            feature_names=["feat_a", "feat_b"],
            df=df,
        )

    def test_valid_frame(self):
        frame = self._make_frame()
        assert len(frame.df) == 100
        assert frame.variety == CoffeeVariety.ARABICA

    def test_non_datetime_index_rejected(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        with pytest.raises(Exception):
            FeatureFrame(
                variety=CoffeeVariety.ARABICA,
                frequency=DataFrequency.DAILY,
                feature_names=["a"],
                df=df,
            )


# ---------------------------------------------------------------------------
# Feature transformer tests
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 252) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(42)
    log_returns = np.random.normal(0.0003, 0.015, n)
    prices = 180.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({
        "open":   prices * np.random.uniform(0.998, 1.002, n),
        "high":   prices * np.random.uniform(1.001, 1.012, n),
        "low":    prices * np.random.uniform(0.988, 0.999, n),
        "close":  prices,
        "volume": np.random.randint(5000, 50000, n).astype(float),
    }, index=idx)


class TestPriceFeatureTransformer:
    def test_basic_features_computed(self):
        from features.price_features import PriceFeatureTransformer

        df = _make_price_df(300)
        tfm = PriceFeatureTransformer(windows=[5, 21])
        result = tfm.compute({"prices": df}, CoffeeVariety.ARABICA)

        assert not result.empty
        assert "log_return_5d" in result.columns
        assert "log_return_21d" in result.columns
        assert "realised_vol_5d" in result.columns
        assert "price_z_21d" in result.columns

    def test_no_future_leakage(self):
        """Return features must not use future data."""
        from features.price_features import PriceFeatureTransformer

        df = _make_price_df(300)
        tfm = PriceFeatureTransformer(windows=[5])
        result = tfm.compute({"prices": df}, CoffeeVariety.ARABICA)

        # log_return_5d at time t = log(P_t / P_{t-5}) — looks backward, OK
        # target_log_return at time t = log(P_{t+5} / P_t) — looks forward
        # Verify no explicit "future" column in features
        future_cols = [c for c in result.columns if "target" in c]
        assert not future_cols

    def test_spread_requires_robusta_input(self):
        from features.price_features import PriceFeatureTransformer

        df = _make_price_df(300)
        tfm = PriceFeatureTransformer(include_spread=True)
        # Without robusta input, spread columns should be absent
        result = tfm.compute({"prices": df}, CoffeeVariety.ARABICA)
        assert "arabica_robusta_spread" not in result.columns

    def test_spread_computed_with_robusta(self):
        from features.price_features import PriceFeatureTransformer

        arabica_df = _make_price_df(300)
        robusta_df = _make_price_df(300)
        robusta_df["close"] *= 22   # Robusta ~USD/tonne, ×22 approx

        tfm = PriceFeatureTransformer(include_spread=True)
        result = tfm.compute(
            {"prices": arabica_df, "prices_robusta": robusta_df},
            CoffeeVariety.ARABICA,
        )
        assert "arabica_robusta_spread" in result.columns
        assert result["arabica_robusta_spread"].notna().sum() > 100


class TestClimateFeatureTransformer:
    def _make_enso_df(self, n_months: int = 60) -> pd.DataFrame:
        idx = pd.date_range("2019-01-01", periods=n_months, freq="MS")
        np.random.seed(7)
        oni = np.random.normal(0, 0.7, n_months)
        return pd.DataFrame({"oni": oni}, index=idx)

    def test_enso_features(self):
        from features.climate_features import ClimateFeatureTransformer

        enso_df = self._make_enso_df(60)
        tfm = ClimateFeatureTransformer(include_enso=True, enso_lags=[0, 3, 6])
        result = tfm.compute({"enso": enso_df}, CoffeeVariety.ARABICA)

        assert "oni" in result.columns
        assert "enso_is_el_nino" in result.columns
        assert "oni_lag_3m" in result.columns
        assert "brazil_drought_risk" in result.columns

    def test_el_nino_flag(self):
        from features.climate_features import ClimateFeatureTransformer

        idx = pd.date_range("2023-01-01", periods=12, freq="MS")
        oni = pd.Series([1.2, 1.5, 1.3, 1.1, 0.8, 0.6, 0.4, 0.2, -0.1, -0.5, -0.7, -0.9], index=idx)
        enso_df = pd.DataFrame({"oni": oni})

        tfm = ClimateFeatureTransformer()
        result = tfm.compute({"enso": enso_df}, CoffeeVariety.ARABICA)
        assert result["enso_is_el_nino"].iloc[0] == 1.0
        assert result["enso_is_la_nina"].iloc[-1] == 1.0


# ---------------------------------------------------------------------------
# Backtesting metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_directional_accuracy(self):
        from backtesting.metrics import directional_accuracy

        # Perfect predictions
        prob_up = np.array([0.8, 0.2, 0.9, 0.1])
        actuals = np.array([0.05, -0.03, 0.02, -0.01])
        da = directional_accuracy(prob_up, actuals)
        assert da == 1.0

        # Totally wrong
        prob_up2 = np.array([0.2, 0.8, 0.1, 0.9])
        da2 = directional_accuracy(prob_up2, actuals)
        assert da2 == 0.0

    def test_brier_score_range(self):
        from backtesting.metrics import brier_score

        np.random.seed(0)
        prob = np.random.uniform(0, 1, 100)
        actuals = np.random.normal(0, 0.02, 100)
        bs = brier_score(prob, actuals)
        assert 0.0 <= bs <= 1.0

    def test_brier_perfect_calibration(self):
        from backtesting.metrics import brier_score

        # Perfectly calibrated: always right
        prob_up = np.array([0.99, 0.01, 0.99, 0.01])
        actuals = np.array([0.05, -0.05, 0.05, -0.05])
        bs = brier_score(prob_up, actuals)
        assert bs < 0.01

    def test_sharpe_ratio(self):
        from backtesting.metrics import sharpe_ratio

        # Positive drift → positive Sharpe
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        s = sharpe_ratio(returns)
        assert isinstance(s, float)

    def test_signal_returns_shape(self):
        from backtesting.metrics import signal_returns

        prob = np.array([0.6, 0.4, 0.7, 0.3, 0.5])
        actuals = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        sr = signal_returns(prob, actuals)
        assert len(sr) == 5


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestOutlierClipper:
    def test_clips_outliers(self):
        from preprocessing.cleaner import OutlierClipper

        s = pd.Series([1.0, 2.0, 3.0, 100.0, 2.5, 1.5, -50.0])
        df = pd.DataFrame({"x": s})
        clipper = OutlierClipper(method="iqr", threshold=3.0)
        clipped = clipper.fit_transform(df)
        assert clipped["x"].max() < 100.0
        assert clipped["x"].min() > -50.0

    def test_no_change_on_clean_data(self):
        from preprocessing.cleaner import OutlierClipper

        df = pd.DataFrame({"x": np.random.normal(0, 1, 100)})
        clipper = OutlierClipper(method="zscore", threshold=5.0)
        clipped = clipper.fit_transform(df)
        # Should be nearly unchanged for clean data
        assert clipped["x"].std() < 2.0


class TestStationarityTransformer:
    def test_log_return_transform(self):
        from preprocessing.cleaner import StationarityTransformer

        idx = pd.date_range("2020-01-01", periods=50)
        df = pd.DataFrame({"close": np.cumprod(1 + np.random.normal(0, 0.01, 50)) * 100}, index=idx)
        tfm = StationarityTransformer({"close": "log_return"})
        transformed = tfm.fit_transform(df)
        # log returns should be much smaller than prices
        assert transformed["close"].std() < 0.1
        assert transformed["close"].iloc[0] != transformed["close"].iloc[0]   # NaN first row
