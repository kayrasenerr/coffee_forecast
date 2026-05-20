"""
models/ensemble.py
==================
Ensemble framework: combines forecasts from multiple models.

Combination strategies:
  equal_weight     — simple average of prob_up across models
  inverse_variance — weight by historical accuracy (1/RMSE)
  regime_switch    — select model by current HMM regime
  stacking         — meta-learner on model outputs (future)

Design:
  - Works with any list of ModelBase instances
  - Optional HMMRegimeDetector for regime-switching
  - Outputs a single ForecastRecord with combined probability
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from contracts.interfaces import ModelBase, RegimeDetectorBase
from contracts.schemas import (
    CoffeeVariety,
    FeatureFrame,
    ForecastRecord,
    MarketRegime,
)

logger = logging.getLogger(__name__)

CombineStrategy = Literal["equal_weight", "inverse_variance", "regime_switch"]


class EnsembleForecaster:
    """
    Combines N ModelBase instances into a single probabilistic forecast.

    Usage:
        ensemble = EnsembleForecaster(
            models={"sarimax": sarimax_model, "bayesian": bayes_model},
            strategy="equal_weight",
        )
        forecast = ensemble.predict(frame, horizon_days=5)
    """

    def __init__(
        self,
        models: Dict[str, ModelBase],
        strategy: CombineStrategy = "equal_weight",
        regime_detector: Optional[RegimeDetectorBase] = None,
        # Regime → preferred model name for regime_switch strategy
        regime_model_map: Optional[Dict[MarketRegime, str]] = None,
        # Per-model weights for inverse_variance (populated after backtest)
        model_weights: Optional[Dict[str, float]] = None,
    ):
        if not models:
            raise ValueError("At least one model required")
        self.models = models
        self.strategy = strategy
        self.regime_detector = regime_detector
        self.regime_model_map = regime_model_map or {}
        self.model_weights = model_weights

    def predict(
        self,
        frame: FeatureFrame,
        horizon_days: int = 5,
    ) -> ForecastRecord:
        """Produce combined probabilistic forecast."""
        individual: Dict[str, ForecastRecord] = {}
        for name, model in self.models.items():
            try:
                individual[name] = model.predict(frame, horizon_days=horizon_days)
            except Exception as exc:
                logger.warning("[Ensemble] Model '%s' failed: %s", name, exc)

        if not individual:
            raise RuntimeError("All ensemble models failed to produce forecasts")

        if self.strategy == "equal_weight":
            return self._combine_equal(individual, horizon_days)
        elif self.strategy == "inverse_variance":
            return self._combine_weighted(individual, horizon_days)
        elif self.strategy == "regime_switch":
            return self._combine_regime(individual, frame, horizon_days)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def individual_forecasts(
        self,
        frame: FeatureFrame,
        horizon_days: int = 5,
    ) -> Dict[str, ForecastRecord]:
        """Return dict of per-model forecasts (useful for diagnostics)."""
        return {
            name: model.predict(frame, horizon_days=horizon_days)
            for name, model in self.models.items()
        }

    # ------------------------------------------------------------------
    # Combination strategies
    # ------------------------------------------------------------------

    def _combine_equal(
        self,
        forecasts: Dict[str, ForecastRecord],
        horizon_days: int,
    ) -> ForecastRecord:
        weights = {name: 1.0 / len(forecasts) for name in forecasts}
        return self._weighted_combine(forecasts, weights, horizon_days)

    def _combine_weighted(
        self,
        forecasts: Dict[str, ForecastRecord],
        horizon_days: int,
    ) -> ForecastRecord:
        if not self.model_weights:
            logger.warning("[Ensemble] No model_weights set; falling back to equal weight")
            return self._combine_equal(forecasts, horizon_days)
        total = sum(self.model_weights.get(n, 1.0) for n in forecasts)
        weights = {n: self.model_weights.get(n, 1.0) / total for n in forecasts}
        return self._weighted_combine(forecasts, weights, horizon_days)

    def _combine_regime(
        self,
        forecasts: Dict[str, ForecastRecord],
        frame: FeatureFrame,
        horizon_days: int,
    ) -> ForecastRecord:
        if self.regime_detector is None:
            return self._combine_equal(forecasts, horizon_days)
        try:
            snapshot = self.regime_detector.predict_latest(frame)
            preferred = self.regime_model_map.get(snapshot.regime)
            if preferred and preferred in forecasts:
                logger.info("[Ensemble] Regime=%s → using model=%s", snapshot.regime.value, preferred)
                return forecasts[preferred]
        except Exception as exc:
            logger.warning("[Ensemble] Regime detection failed: %s", exc)
        return self._combine_equal(forecasts, horizon_days)

    @staticmethod
    def _weighted_combine(
        forecasts: Dict[str, ForecastRecord],
        weights: Dict[str, float],
        horizon_days: int,
    ) -> ForecastRecord:
        """Blend forecasts using weighted average of key probabilistic quantities."""
        mean_ret = sum(weights[n] * f.mean_return for n, f in forecasts.items())
        prob_up  = sum(weights[n] * f.prob_up   for n, f in forecasts.items())
        q10 = sum(weights[n] * f.q10 for n, f in forecasts.items())
        q25 = sum(weights[n] * f.q25 for n, f in forecasts.items())
        q75 = sum(weights[n] * f.q75 for n, f in forecasts.items())
        q90 = sum(weights[n] * f.q90 for n, f in forecasts.items())
        prob_large = sum(weights[n] * f.prob_large_move for n, f in forecasts.items())

        # Use variety from first model
        first = next(iter(forecasts.values()))
        model_names = "+".join(forecasts.keys())

        return ForecastRecord(
            generated_at=pd.Timestamp.utcnow().to_pydatetime(),
            forecast_horizon_days=horizon_days,
            variety=first.variety,
            mean_return=round(mean_ret, 6),
            median_return=round(mean_ret, 6),
            q10=round(q10, 6),
            q25=round(q25, 6),
            q75=round(q75, 6),
            q90=round(q90, 6),
            prob_up=round(prob_up, 4),
            prob_large_move=round(prob_large, 4),
            model_name=f"ensemble[{model_names}]",
        )
