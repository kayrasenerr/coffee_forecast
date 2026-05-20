"""
models/garch_model.py
=====================
GARCH family volatility models for coffee futures.

Coffee exhibits strong volatility clustering:
  - Low-vol regimes (normal trade)
  - High-vol spikes (frost scares, crop reports, geopolitical events)
  - Persistent elevated vol during supply shocks

Model choices:
  GARCH(1,1)   — baseline, well-calibrated for commodities
  EGARCH(1,1)  — captures asymmetric vol (bad news > good news)
  GJR-GARCH    — alternative asymmetric specification

Distribution:
  Student-t    — fat tails in commodity returns

Library: arch (Kevin Sheppard's ARCH library)

Outputs:
  - Conditional volatility series
  - N-day-ahead vol forecast
  - Volatility regime classification (low/medium/high/extreme)
  - Annualised vol forecast
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from contracts.interfaces import VolatilityModelBase
from contracts.schemas import CoffeeVariety, VolatilityForecast
from config.settings import settings

logger = logging.getLogger(__name__)

# Volatility regime thresholds (annualised, for coffee)
_VOL_THRESHOLDS = {
    "low":     0.20,   # < 20% annualised vol
    "medium":  0.30,   # 20–30%
    "high":    0.45,   # 30–45%
    "extreme": float("inf"),  # > 45%
}


class GARCHVolatilityModel(VolatilityModelBase):
    """
    GARCH-family conditional volatility model.

    Usage:
        model = GARCHVolatilityModel(p=1, q=1, dist="t")
        model.fit(returns)
        forecast = model.forecast(returns, horizon_days=10)
    """

    model_name = "garch"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        model_type: str = "GARCH",     # "GARCH" | "EGARCH" | "GJR-GARCH"
        dist: str = "t",               # "normal" | "t" | "skewt"
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        annualise_days: int = 252,
    ):
        self.p = p
        self.q = q
        self.model_type = model_type
        self.dist = dist
        self.variety = variety
        self.annualise_days = annualise_days

        self._result = None
        self._fitted = False

    def fit(self, returns: pd.Series) -> "GARCHVolatilityModel":
        """
        Fit GARCH model on log-return series.

        Parameters
        ----------
        returns : pd.Series of log returns (daily)
        """
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("pip install arch") from e

        clean_returns = returns.dropna() * 100   # arch expects percentage returns

        logger.info(
            "[GARCH] Fitting %s(%d,%d) with %s dist on %d obs",
            self.model_type, self.p, self.q, self.dist, len(clean_returns)
        )

        model = arch_model(
            clean_returns,
            vol=self.model_type,
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean="Constant",
            rescale=False,
        )
        self._result = model.fit(
            update_freq=0,
            disp="off",
            show_warning=False,
        )
        self._fitted = True
        logger.info("[GARCH] AIC: %.2f  BIC: %.2f", self._result.aic, self._result.bic)
        return self

    def forecast(
        self,
        returns: pd.Series,
        horizon_days: int = 10,
    ) -> VolatilityForecast:
        """Generate N-day-ahead volatility forecast."""
        self._check_fitted()

        # Re-fit on latest returns for rolling application
        self.fit(returns)

        fcast = self._result.forecast(
            horizon=horizon_days,
            reindex=False,
            method="simulation",
        )
        # Variance forecasts (percentage returns squared)
        variance_forecast = fcast.variance.iloc[-1].values  # shape: (horizon,)
        vol_daily_pct = np.sqrt(variance_forecast)
        vol_annual = vol_daily_pct / 100 * np.sqrt(self.annualise_days)

        # Mean forecast vol over horizon
        mean_forecast_vol = float(vol_annual.mean())

        # Current conditional vol
        current_vol = float(
            np.sqrt(self._result.conditional_volatility.iloc[-1]) / 100 * np.sqrt(self.annualise_days)
        )

        vol_regime = self._classify_vol(mean_forecast_vol)

        return VolatilityForecast(
            generated_at=pd.Timestamp.now('UTC').replace(tzinfo=None),
            variety=self.variety,
            horizon_days=horizon_days,
            forecast_annualized_vol=round(mean_forecast_vol, 4),
            current_annualized_vol=round(current_vol, 4),
            vol_regime=vol_regime,
            model_name=f"{self.model_type}({self.p},{self.q})-{self.dist}",
        )

    def conditional_volatility(self) -> pd.Series:
        """Return the in-sample conditional volatility series (annualised)."""
        self._check_fitted()
        raw = self._result.conditional_volatility
        if raw is None or len(raw) == 0:
            return pd.Series(dtype=float)
        cond_vol = raw / 100 * np.sqrt(self.annualise_days)
        return cond_vol

    def model_summary(self) -> str:
        """Return arch model summary as string."""
        self._check_fitted()
        return self._result.summary().as_text()

    def residual_diagnostics(self) -> dict:
        """
        Basic residual diagnostics:
          - Ljung-Box test on standardised residuals
          - ARCH-LM test on squared residuals
        """
        self._check_fitted()
        std_resid = self._result.std_resid.dropna()
        if len(std_resid) < 20:
            return {"ljung_box_resid_pvalue": float("nan"),
                    "ljung_box_resid_sq_pvalue": float("nan"),
                    "no_remaining_arch": None}
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb  = acorr_ljungbox(std_resid,    lags=[10], return_df=True)
            lb2 = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
            return {
                "ljung_box_resid_pvalue":    float(lb["lb_pvalue"].iloc[0]),
                "ljung_box_resid_sq_pvalue": float(lb2["lb_pvalue"].iloc[0]),
                "no_remaining_arch": float(lb2["lb_pvalue"].iloc[0]) > 0.05,
            }
        except Exception as exc:
            return {"ljung_box_resid_pvalue": float("nan"),
                    "ljung_box_resid_sq_pvalue": float("nan"),
                    "no_remaining_arch": None, "diag_error": str(exc)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "result_params": self._result.params if self._result else None,
            "result_resid": self._result.resid if self._result else None,
            "p": self.p, "q": self.q,
            "model_type": self.model_type,
            "dist": self.dist,
            "variety": self.variety.value,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> "GARCHVolatilityModel":
        # Full model restoration requires re-fitting; we load params only
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.p = state["p"]
        self.q = state["q"]
        self.model_type = state["model_type"]
        self.dist = state["dist"]
        self.variety = CoffeeVariety(state["variety"])
        logger.warning("[GARCH] Loaded metadata only — call fit() to restore full model")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_vol(annualised_vol: float) -> str:
        for regime, threshold in _VOL_THRESHOLDS.items():
            if annualised_vol < threshold:
                return regime
        return "extreme"

    def _check_fitted(self) -> None:
        if not self._fitted or self._result is None:
            raise RuntimeError("Call fit() before using this method")
