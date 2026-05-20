"""
models/sarimax_model.py
=======================
SARIMAX model for directional coffee price forecasting.

SARIMAX:
  - AR component captures momentum / mean reversion
  - MA component captures shock propagation
  - Seasonal component captures harvest cycles (52-week for weekly)
  - Exogenous variables: ENSO, BRL, COT positioning, volatility

Outputs:
  - N-step-ahead conditional mean forecast
  - Confidence intervals → converted to probabilistic forecast
  - Directional probability (P(return > 0))

Note: SARIMAX assumes stationarity. Feed log returns (already stationary),
not prices.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from contracts.interfaces import ModelBase
from contracts.schemas import (
    CoffeeVariety,
    DataFrequency,
    FeatureFrame,
    ForecastRecord,
    MarketRegime,
)

logger = logging.getLogger(__name__)

# Exogenous features to use (subset of FeatureFrame available columns)
_DEFAULT_EXOG_FEATURES = [
    "oni",                      # ENSO
    "brl_farmer_incentive",     # USD/BRL
    "cot_net_noncomm_norm",     # COT positioning
    "realised_vol_21d",         # Volatility state
    "arabica_robusta_spread",   # Inter-market spread
]


class SARIMAXForecaster(ModelBase):
    """
    SARIMAX forecaster for coffee returns.

    Usage:
        model = SARIMAXForecaster(
            order=(1,0,1),
            variety=CoffeeVariety.ARABICA,
            exog_features=["oni", "brl_farmer_incentive"]
        )
        model.fit(frame, target_col="log_return_1d")
        forecast = model.predict(frame, horizon_days=5)
    """

    model_name = "sarimax"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        exog_features: Optional[List[str]] = None,
        large_move_threshold: float = 0.02,   # 2% = "large move"
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.variety = variety
        self.exog_features = exog_features or _DEFAULT_EXOG_FEATURES
        self.large_move_threshold = large_move_threshold

        self._result = None
        self._fitted_exog_cols: List[str] = []
        self._fitted = False

    def fit(self, frame: FeatureFrame, target_col: str) -> "SARIMAXForecaster":
        """Fit SARIMAX on FeatureFrame."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as e:
            raise ImportError("pip install statsmodels") from e

        df = frame.df.copy()
        target = df[target_col].dropna()

        # Select available exog columns
        available_exog = [c for c in self.exog_features if c in df.columns]
        self._fitted_exog_cols = available_exog

        exog = None
        if available_exog:
            exog = df[available_exog].reindex(target.index)
            # Forward fill any gaps in exog
            exog = exog.ffill().bfill()
            # Drop rows where exog is still NaN
            valid_mask = exog.notna().all(axis=1)
            target = target[valid_mask]
            exog = exog[valid_mask]

        logger.info(
            "[SARIMAX] Fitting ARIMA%s x SARIMA%s on %d obs, %d exog vars",
            self.order, self.seasonal_order, len(target), len(available_exog)
        )

        model = SARIMAX(
            endog=target,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(
            disp=False,
            maxiter=200,
            method="lbfgs",
        )
        self._fitted = True
        logger.info(
            "[SARIMAX] AIC: %.2f  BIC: %.2f",
            self._result.aic, self._result.bic
        )
        return self

    def predict(
        self,
        frame: FeatureFrame,
        horizon_days: int = 5,
    ) -> ForecastRecord:
        """Generate probabilistic forecast for next `horizon_days`."""
        self._check_fitted()

        exog_forecast = None
        if self._fitted_exog_cols:
            latest_exog = (
                frame.df[self._fitted_exog_cols]
                .iloc[-horizon_days:]
                .ffill()
                .fillna(0)   # fallback for missing
            )
            exog_forecast = latest_exog.values

        try:
            forecast_obj = self._result.get_forecast(
                steps=horizon_days,
                exog=exog_forecast,
            )
            mean_forecast = forecast_obj.predicted_mean
            ci = forecast_obj.conf_int(alpha=0.10)   # 90% CI

            mean_ret = float(mean_forecast.mean())
            ci_low = float(ci.iloc[:, 0].mean())
            ci_high = float(ci.iloc[:, 1].mean())

        except Exception as exc:
            logger.warning("[SARIMAX] Forecast failed: %s. Falling back to last values.", exc)
            mean_ret = 0.0
            ci_low = -self.large_move_threshold
            ci_high = self.large_move_threshold

        # Derive probabilistic estimates assuming normal residuals
        resid_std = self._residual_std()
        cumulative_std = resid_std * np.sqrt(horizon_days)

        prob_up = float(1 - stats.norm.cdf(0, loc=mean_ret, scale=cumulative_std))
        prob_large_move = float(
            stats.norm.sf(self.large_move_threshold, loc=mean_ret, scale=cumulative_std)
            + stats.norm.cdf(-self.large_move_threshold, loc=mean_ret, scale=cumulative_std)
        )

        return ForecastRecord(
            generated_at=pd.Timestamp.now('UTC').replace(tzinfo=None),
            forecast_horizon_days=horizon_days,
            variety=self.variety,
            mean_return=round(mean_ret, 6),
            median_return=round(mean_ret, 6),
            q10=round(stats.norm.ppf(0.10, loc=mean_ret, scale=cumulative_std), 6),
            q25=round(stats.norm.ppf(0.25, loc=mean_ret, scale=cumulative_std), 6),
            q75=round(stats.norm.ppf(0.75, loc=mean_ret, scale=cumulative_std), 6),
            q90=round(stats.norm.ppf(0.90, loc=mean_ret, scale=cumulative_std), 6),
            prob_up=round(prob_up, 4),
            prob_large_move=round(prob_large_move, 4),
            model_name=self.model_name,
        )

    def in_sample_diagnostics(self) -> dict:
        """Return diagnostic statistics for model validation."""
        self._check_fitted()
        resid = self._result.resid
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
        return {
            "aic": round(self._result.aic, 2),
            "bic": round(self._result.bic, 2),
            "log_likelihood": round(self._result.llf, 2),
            "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
            "no_serial_correlation": float(lb["lb_pvalue"].iloc[0]) > 0.05,
            "resid_mean": round(resid.mean(), 6),
            "resid_std": round(resid.std(), 6),
        }

    def get_params(self) -> dict:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_features": self.exog_features,
            "large_move_threshold": self.large_move_threshold,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "result": self._result,
                "exog_cols": self._fitted_exog_cols,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "variety": self.variety.value,
            }, f)
        logger.info("[SARIMAX] Saved to %s", path)

    def load(self, path: str) -> "SARIMAXForecaster":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._result = state["result"]
        self._fitted_exog_cols = state["exog_cols"]
        self.order = state["order"]
        self.seasonal_order = state["seasonal_order"]
        self.variety = CoffeeVariety(state["variety"])
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _residual_std(self) -> float:
        if self._result is None:
            return 0.01
        return float(self._result.resid.std())

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
