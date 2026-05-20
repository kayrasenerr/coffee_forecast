"""
models/forecast.py
SARIMAX directional forecast for Arabica log returns.

Exogenous variables (when available):
  - USD/BRL log return lag-1 (causal: Brazil is ~40% of supply)
  - Realised volatility
  - Return z-score

For OOS forecast without known future exog, we use persistence:
last observed exog values are carried forward. This is clearly labelled
in the output. Replace with proper forecasts of exog when available.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config.settings import ModelConfig
from schemas.types import FeatureMatrix, ForecastResult

EXOG_COLS = [
    "price_fx_usd_brl_ret_lag1",
    "vol_realised_vol",
    "price_zscore_ret",
]


class SARIMAXForecastModel:
    model_name = "SARIMAX"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._fit_result = None
        self._exog_cols: list = []
        self._last_exog: pd.DataFrame | None = None   # for OOS persistence

    def _select_exog(self, features: pd.DataFrame):
        available = [c for c in EXOG_COLS if c in features.columns]
        self._exog_cols = available
        return features[available] if available else None

    def fit(self, fm: FeatureMatrix) -> "SARIMAXForecastModel":
        endog = fm.target
        exog  = self._select_exog(fm.features)
        if exog is not None:
            self._last_exog = exog.iloc[[-1]]   # save last row for OOS persistence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog, exog=exog,
                order=self.cfg.sarimax_order,
                seasonal_order=self.cfg.sarimax_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fit_result = model.fit(disp=False, maxiter=200)
        return self

    def predict(self, fm: FeatureMatrix) -> ForecastResult:
        """In-sample / in-period predictions (used by walk-forward)."""
        if self._fit_result is None:
            raise RuntimeError("Call fit() first")
        exog = self._select_exog(fm.features) if self._exog_cols else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._fit_result.get_prediction(
                start=fm.features.index[0],
                end=fm.features.index[-1],
                exog=exog,
            )
        summary = pred.summary_frame(alpha=0.2)
        idx = fm.features.index
        return ForecastResult(
            symbol=fm.symbol,
            forecast_dates=idx,
            mean=summary["mean"].reindex(idx),
            lower=summary["mean_ci_lower"].reindex(idx),
            upper=summary["mean_ci_upper"].reindex(idx),
            model_name=self.model_name,
            horizon=self.cfg.forecast_horizon,
        )

    def forecast_oos(self, steps: int, exog_future: pd.DataFrame | None = None) -> ForecastResult:
        """
        True OOS forecast.
        If exog_future is None and model has exog, uses persistence (last known values).
        Persistence is a naive but honest baseline — clearly documented.
        """
        if self._fit_result is None:
            raise RuntimeError("Call fit() first")

        future_exog = None
        if self._exog_cols:
            if exog_future is not None:
                future_exog = exog_future[self._exog_cols].values
            elif self._last_exog is not None:
                # Repeat last known exog (persistence assumption)
                future_exog = np.tile(self._last_exog[self._exog_cols].values, (steps, 1))
            else:
                future_exog = np.zeros((steps, len(self._exog_cols)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._fit_result.get_forecast(steps=steps, exog=future_exog)

        summary = fc.summary_frame(alpha=0.2)
        return ForecastResult(
            symbol="arabica",
            forecast_dates=summary.index,
            mean=summary["mean"],
            lower=summary["mean_ci_lower"],
            upper=summary["mean_ci_upper"],
            model_name=self.model_name,
            horizon=steps,
        )

    def fit_predict(self, fm: FeatureMatrix) -> ForecastResult:
        return self.fit(fm).predict(fm)
