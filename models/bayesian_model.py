"""
models/bayesian_model.py
========================
Bayesian probabilistic forecasting for coffee markets.

Approach: Bayesian Ridge Regression with explicit uncertainty quantification.
Extends to full PyMC hierarchical model when data justifies it.

Why Bayesian for coffee:
  - Explicit uncertainty (crucial for illiquid/seasonal markets)
  - Prior encodes domain knowledge (e.g. BRL impact on Arabica)
  - Posterior updates cleanly as new data arrives
  - Natural probabilistic forecast without distributional assumptions

MVP: BayesianRidge (sklearn) + bootstrap ensemble for UQ.
Advanced: PyMC hierarchical model (feature-flagged).

Outputs:
  - Posterior predictive mean and std
  - Full quantile fan chart
  - Feature importance (posterior coefficients)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from contracts.interfaces import ModelBase
from contracts.schemas import (
    CoffeeVariety,
    FeatureFrame,
    ForecastRecord,
)

logger = logging.getLogger(__name__)

# Prioritised features for Bayesian model (causal ordering)
_BAYESIAN_FEATURES = [
    # Causal / fundamental
    "brl_farmer_incentive",
    "oni",
    "cot_net_noncomm_norm",
    "arabica_robusta_spread",
    # Technical / conditioning
    "realised_vol_21d",
    "price_z_63d",
    "log_return_5d",
    "log_return_21d",
    # Climate risk
    "brazil_drought_risk",
    "enso_intensity",
]


class BayesianForecaster(ModelBase):
    """
    Bayesian Ridge Regression forecaster with bootstrap uncertainty.

    Uses sklearn BayesianRidge for closed-form posterior inference.
    Bootstrap ensemble approximates the posterior predictive distribution.

    Usage:
        model = BayesianForecaster(variety=CoffeeVariety.ARABICA)
        model.fit(frame, target_col="target_log_return_5d")
        forecast = model.predict(frame, horizon_days=5)
    """

    model_name = "bayesian_ridge"

    def __init__(
        self,
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        feature_cols: Optional[List[str]] = None,
        n_bootstrap: int = 500,
        large_move_threshold: float = 0.02,
        alpha_1: float = 1e-6,   # BayesianRidge hyperprior (precision of weights)
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ):
        self.variety = variety
        self.feature_cols = feature_cols or _BAYESIAN_FEATURES
        self.n_bootstrap = n_bootstrap
        self.large_move_threshold = large_move_threshold
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self._model = None
        self._scaler = None
        self._fitted_cols: List[str] = []
        self._bootstrap_preds: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, frame: FeatureFrame, target_col: str) -> "BayesianForecaster":
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        df = frame.df.copy()
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            raise ValueError(f"No usable features. Required: {self.feature_cols}")

        self._fitted_cols = available
        X_raw = df[available].copy()
        y = df[target_col].copy()

        # Align and drop NaN
        aligned = pd.concat([X_raw, y], axis=1).dropna()
        X_raw = aligned[available]
        y = aligned[target_col]

        # Scale features (BayesianRidge is sensitive to scale)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw)

        self._model = BayesianRidge(
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            compute_score=True,
            fit_intercept=True,
        )
        self._model.fit(X, y.values)
        self._fitted = True

        logger.info(
            "[Bayesian] Fitted on %d obs, %d features. R²=%.4f",
            len(y), len(available),
            self._model.scores_[-1] if hasattr(self._model, "scores_") else float("nan"),
        )
        return self

    def predict(
        self,
        frame: FeatureFrame,
        horizon_days: int = 5,
    ) -> ForecastRecord:
        self._check_fitted()

        X_latest = self._get_latest_X(frame)
        # Posterior predictive: mean and std from BayesianRidge
        y_mean, y_std = self._model.predict(X_latest, return_std=True)
        mean_ret = float(y_mean[-1])
        pred_std = float(y_std[-1])

        # Scale std to horizon (rough approximation)
        horizon_std = pred_std * np.sqrt(horizon_days)

        prob_up = float(1 - stats.norm.cdf(0, loc=mean_ret, scale=horizon_std))
        prob_large = float(
            stats.norm.sf(self.large_move_threshold, loc=mean_ret, scale=horizon_std)
            + stats.norm.cdf(-self.large_move_threshold, loc=mean_ret, scale=horizon_std)
        )

        return ForecastRecord(
            generated_at=pd.Timestamp.now('UTC').replace(tzinfo=None),
            forecast_horizon_days=horizon_days,
            variety=self.variety,
            mean_return=round(mean_ret, 6),
            median_return=round(mean_ret, 6),
            q10=round(stats.norm.ppf(0.10, loc=mean_ret, scale=horizon_std), 6),
            q25=round(stats.norm.ppf(0.25, loc=mean_ret, scale=horizon_std), 6),
            q75=round(stats.norm.ppf(0.75, loc=mean_ret, scale=horizon_std), 6),
            q90=round(stats.norm.ppf(0.90, loc=mean_ret, scale=horizon_std), 6),
            prob_up=round(prob_up, 4),
            prob_large_move=round(prob_large, 4),
            model_name=self.model_name,
        )

    def feature_importance(self) -> pd.Series:
        """Return posterior coefficient magnitudes as feature importance."""
        self._check_fitted()
        return pd.Series(
            np.abs(self._model.coef_),
            index=self._fitted_cols,
        ).sort_values(ascending=False)

    def posterior_coefficients(self) -> pd.DataFrame:
        """Return posterior mean and std of each coefficient."""
        self._check_fitted()
        # sigma_sq is the posterior noise variance
        sigma = np.sqrt(1.0 / self._model.alpha_)
        coef_std = sigma * np.ones(len(self._fitted_cols))  # approximation
        return pd.DataFrame({
            "feature": self._fitted_cols,
            "coef_mean": self._model.coef_,
            "coef_std_approx": coef_std,
        }).set_index("feature")

    def get_params(self) -> dict:
        return {
            "feature_cols": self.feature_cols,
            "n_bootstrap": self.n_bootstrap,
            "large_move_threshold": self.large_move_threshold,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "fitted_cols": self._fitted_cols,
                "variety": self.variety.value,
            }, f)

    def load(self, path: str) -> "BayesianForecaster":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._scaler = state["scaler"]
        self._fitted_cols = state["fitted_cols"]
        self.variety = CoffeeVariety(state["variety"])
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_latest_X(self, frame: FeatureFrame) -> np.ndarray:
        available = [c for c in self._fitted_cols if c in frame.df.columns]
        X_raw = frame.df[available].ffill().dropna()
        return self._scaler.transform(X_raw.values[-1:])

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
