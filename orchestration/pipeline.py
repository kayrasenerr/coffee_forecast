"""
orchestration/pipeline.py
=========================
Top-level pipeline orchestrator for the Coffee Quant system.

This is the entry point for running the full system:
  ingest → preprocess → feature engineering → fit → backtest → report

Responsibilities:
  - Coordinates all modules via their public interfaces
  - Reads configuration from settings / YAML
  - Manages experiment tracking
  - Produces artefacts (model files, plots, metrics)
  - Designed to run as a scheduled job or ad-hoc CLI call

Usage:
    from orchestration.pipeline import CoffeeQuantPipeline
    from contracts.schemas import CoffeeVariety

    pipeline = CoffeeQuantPipeline()
    pipeline.run(variety=CoffeeVariety.ARABICA, backtest=True, plots=True)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.settings import settings, ensure_dirs
from contracts.schemas import CoffeeVariety, DataFrequency
from features.pipeline import FeaturePipeline
from features.store import ParquetFeatureStore
from experiment.tracker import FileExperimentTracker

logger = logging.getLogger(__name__)


class CoffeeQuantPipeline:
    """
    Orchestrates a full research run for one variety.

    Modular design: each phase is a separate method so it can be
    called independently during development.
    """

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        ensure_dirs()
        self.start_date = date.fromisoformat(start_date or settings.default_start_date)
        self.end_date   = date.fromisoformat(end_date   or settings.default_end_date)
        self.store   = ParquetFeatureStore(settings.features_dir)
        self.tracker = FileExperimentTracker(settings.experiments_dir)

    # ------------------------------------------------------------------
    # Phase 1: Ingestion
    # ------------------------------------------------------------------

    def ingest(self, variety: CoffeeVariety) -> Dict[str, pd.DataFrame]:
        """Fetch all enabled data sources for a variety."""
        from ingestion.registry import source_registry

        raw: Dict[str, pd.DataFrame] = {}
        source_ids = source_registry.list_enabled()
        logger.info("[Pipeline] Ingesting %d sources: %s", len(source_ids), source_ids)

        for sid in source_ids:
            try:
                source = source_registry.get(sid)
                df = source.fetch_validated(self.start_date, self.end_date)
                if not df.empty:
                    raw[sid] = df
                    logger.info("[Pipeline] %s: %d rows", sid, len(df))
                else:
                    logger.warning("[Pipeline] %s: empty response", sid)
            except Exception as exc:
                logger.error("[Pipeline] Ingestion failed for %s: %s", sid, exc)

        return raw

    # ------------------------------------------------------------------
    # Phase 2: Feature Engineering
    # ------------------------------------------------------------------

    def build_features(
        self,
        raw: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ):
        """Run feature pipeline and save to store."""
        store_name = f"{variety.value}_features_{frequency.value}"
        fp = FeaturePipeline(store=self.store)
        frame = fp.run(
            raw_inputs=raw,
            variety=variety,
            frequency=frequency,
            store_name=store_name,
        )
        logger.info("[Pipeline] Feature frame saved as '%s'", store_name)
        return frame

    # ------------------------------------------------------------------
    # Phase 3: Model Fitting
    # ------------------------------------------------------------------

    def fit_models(self, frame, variety: CoffeeVariety, run_id: str) -> dict:
        """Fit HMM, GARCH, SARIMAX, and Bayesian models."""
        from models.hmm_model import HMMRegimeDetector
        from models.garch_model import GARCHVolatilityModel
        from models.sarimax_model import SARIMAXForecaster
        from models.bayesian_model import BayesianForecaster

        models_dir = settings.models_dir / run_id
        models_dir.mkdir(parents=True, exist_ok=True)
        fitted = {}

        # --- HMM Regime Detector ---
        try:
            hmm = HMMRegimeDetector(
                n_regimes=settings.hmm_n_regimes,
                variety=variety,
                covariance_type=settings.hmm_covariance_type,
                n_iter=settings.hmm_n_iter,
            )
            hmm.fit(frame)
            hmm.save(str(models_dir / "hmm.pkl"))
            fitted["hmm"] = hmm
            latest_regime = hmm.predict_latest(frame)
            self.tracker.log_metrics(run_id, {
                "hmm_current_regime_id": latest_regime.regime_id,
                "hmm_regime_confidence": latest_regime.probability,
            })
            logger.info("[Pipeline] HMM fitted. Current regime: %s (p=%.3f)",
                        latest_regime.regime.value, latest_regime.probability)
        except Exception as exc:
            logger.error("[Pipeline] HMM failed: %s", exc)

        # --- GARCH Volatility ---
        try:
            if "log_return_1d" in frame.df.columns:
                garch = GARCHVolatilityModel(
                    p=settings.garch_p,
                    q=settings.garch_q,
                    dist=settings.garch_dist,
                    variety=variety,
                )
                returns = frame.df["log_return_1d"].dropna()
                garch.fit(returns)
                vol_fc = garch.forecast(returns, horizon_days=10)
                garch.save(str(models_dir / "garch.pkl"))
                fitted["garch"] = garch
                self.tracker.log_metrics(run_id, {
                    "garch_current_vol": vol_fc.current_annualized_vol,
                    "garch_forecast_vol": vol_fc.forecast_annualized_vol,
                })
                logger.info("[Pipeline] GARCH fitted. Forecast vol: %.1f%% (%s)",
                            vol_fc.forecast_annualized_vol * 100, vol_fc.vol_regime)
        except Exception as exc:
            logger.error("[Pipeline] GARCH failed: %s", exc)

        # --- SARIMAX ---
        try:
            sarimax = SARIMAXForecaster(
                order=settings.sarimax_order,
                variety=variety,
            )
            sarimax.fit(frame, target_col="target_log_return_5d")
            sarimax.save(str(models_dir / "sarimax.pkl"))
            fitted["sarimax"] = sarimax
            diag = sarimax.in_sample_diagnostics()
            self.tracker.log_metrics(run_id, {
                "sarimax_aic": diag["aic"],
                "sarimax_bic": diag["bic"],
            })
            logger.info("[Pipeline] SARIMAX fitted. AIC=%.1f", diag["aic"])
        except Exception as exc:
            logger.error("[Pipeline] SARIMAX failed: %s", exc)

        # --- Bayesian ---
        try:
            bayes = BayesianForecaster(variety=variety)
            bayes.fit(frame, target_col="target_log_return_5d")
            bayes.save(str(models_dir / "bayesian.pkl"))
            fitted["bayesian"] = bayes
            logger.info("[Pipeline] Bayesian fitted.")
        except Exception as exc:
            logger.error("[Pipeline] Bayesian failed: %s", exc)

        return fitted

    # ------------------------------------------------------------------
    # Phase 4: Backtesting
    # ------------------------------------------------------------------

    def run_backtest(self, frame, fitted_models: dict, run_id: str) -> dict:
        """Walk-forward backtest all directional models."""
        from backtesting.walk_forward import WalkForwardBacktester

        backtester = WalkForwardBacktester(
            train_window=settings.backtest_train_window_days,
            test_window=settings.backtest_test_window_days,
            step=settings.backtest_step_days,
        )
        results = {}

        for name in ["sarimax", "bayesian"]:
            if name not in fitted_models:
                continue
            try:
                logger.info("[Pipeline] Backtesting %s …", name)
                result = backtester.run(
                    model=fitted_models[name],
                    frame=frame,
                    target_col="target_log_return_5d",
                    n_folds=settings.backtest_n_folds,
                )
                results[name] = result
                self.tracker.log_metrics(run_id, {
                    f"{name}_da": result.directional_accuracy,
                    f"{name}_brier": result.mean_brier_score or float("nan"),
                    f"{name}_sharpe": result.signal_sharpe or float("nan"),
                })
                logger.info(
                    "[Pipeline] %s backtest: DA=%.3f  Brier=%.4f  Sharpe=%.2f",
                    name,
                    result.directional_accuracy,
                    result.mean_brier_score or float("nan"),
                    result.signal_sharpe or float("nan"),
                )
            except Exception as exc:
                logger.error("[Pipeline] Backtest %s failed: %s", name, exc)

        return results

    # ------------------------------------------------------------------
    # Phase 5: Forecast
    # ------------------------------------------------------------------

    def generate_forecast(self, frame, fitted_models: dict, run_id: str) -> dict:
        """Generate latest forecasts from all models."""
        from models.ensemble import EnsembleForecaster

        forecasts = {}
        for name, model in fitted_models.items():
            if name in ("hmm", "garch"):
                continue
            try:
                fc = model.predict(frame, horizon_days=5)
                forecasts[name] = fc
                logger.info(
                    "[Pipeline] %s forecast: prob_up=%.2f  mean_ret=%.4f",
                    name, fc.prob_up, fc.mean_return
                )
                self.tracker.log_metrics(run_id, {
                    f"{name}_prob_up": fc.prob_up,
                    f"{name}_mean_return": fc.mean_return,
                })
            except Exception as exc:
                logger.error("[Pipeline] Forecast %s failed: %s", name, exc)

        # Ensemble
        if len(forecasts) >= 2:
            try:
                ensemble = EnsembleForecaster(
                    models={k: v for k, v in fitted_models.items() if k in forecasts},
                    regime_detector=fitted_models.get("hmm"),
                )
                ensemble_fc = ensemble.predict(frame, horizon_days=5)
                forecasts["ensemble"] = ensemble_fc
                logger.info("[Pipeline] Ensemble: prob_up=%.2f", ensemble_fc.prob_up)
            except Exception as exc:
                logger.error("[Pipeline] Ensemble failed: %s", exc)

        return forecasts

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        frequency: DataFrequency = DataFrequency.DAILY,
        backtest: bool = True,
        plots: bool = False,
    ) -> dict:
        """Execute the complete research pipeline."""
        run_id = self.tracker.start_run(
            f"{variety.value}_full_run",
            tags={"variety": variety.value, "frequency": frequency.value},
        )
        self.tracker.log_params(run_id, {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "variety": variety.value,
            "frequency": frequency.value,
        })

        logger.info("=" * 60)
        logger.info("Coffee Quant Pipeline — %s", variety.value.upper())
        logger.info("Period: %s → %s", self.start_date, self.end_date)
        logger.info("=" * 60)

        raw     = self.ingest(variety)
        frame   = self.build_features(raw, variety, frequency)
        fitted  = self.fit_models(frame, variety, run_id)

        bt_results = {}
        if backtest:
            bt_results = self.run_backtest(frame, fitted, run_id)

        forecasts = self.generate_forecast(frame, fitted, run_id)

        if plots:
            self._generate_plots(frame, fitted, run_id)

        self.tracker.end_run(run_id)
        logger.info("[Pipeline] Run complete. ID: %s", run_id)

        return {
            "run_id": run_id,
            "frame": frame,
            "models": fitted,
            "backtest_results": bt_results,
            "forecasts": forecasts,
        }

    def _generate_plots(self, frame, fitted_models: dict, run_id: str) -> None:
        """Generate and save all standard plots."""
        from visualization.plots import (
            plot_regime_overlay, plot_volatility_bands,
            plot_feature_importance, save_figure,
        )
        plots_dir = settings.experiments_dir / run_id / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        close = frame.df.get("close") or frame.df.iloc[:, 0]

        # Regime overlay
        if "hmm" in fitted_models:
            try:
                snapshots = fitted_models["hmm"].predict(frame)
                regime_series = pd.Series(
                    {s.timestamp: s.regime.value for s in snapshots}
                )
                fig = plot_regime_overlay(close, regime_series)
                save_figure(fig, str(plots_dir / "regime_overlay.png"))
            except Exception as exc:
                logger.warning("[Pipeline] Regime plot failed: %s", exc)

        # Feature importance
        if "bayesian" in fitted_models:
            try:
                importance = fitted_models["bayesian"].feature_importance()
                fig = plot_feature_importance(importance)
                save_figure(fig, str(plots_dir / "feature_importance.png"))
            except Exception as exc:
                logger.warning("[Pipeline] Feature importance plot failed: %s", exc)
