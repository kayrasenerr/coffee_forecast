"""
models/volatility.py
GARCH(1,1) conditional volatility model for Arabica log returns.

Returns:
  - Fitted conditional volatility series (in-sample)
  - One-step-ahead volatility forecast
  - Annualised vol in consistent units

Why GARCH for coffee:
  - Coffee vol is highly clustered (frost scares, La Niña events)
  - GARCH captures the persistence of stress periods
  - Volatility forecast feeds regime-transition probability
"""
import numpy as np
import pandas as pd
from arch import arch_model
from config.settings import ModelConfig
from schemas.types import VolatilityResult


class GARCHVolatilityModel:
    model_name = "GARCH(1,1)"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._result = None

    def fit(self, log_returns: pd.Series) -> "GARCHVolatilityModel":
        """
        Args:
            log_returns: daily log return series (not annualised)
        """
        # arch_model wants returns in % for numerical stability
        scaled = log_returns.dropna() * 100
        model = arch_model(
            scaled,
            vol="Garch",
            p=self.cfg.garch_p,
            q=self.cfg.garch_q,
            dist="normal",
            rescale=False,
        )
        self._result = model.fit(disp="off", show_warning=False)
        return self

    def get_result(self, log_returns: pd.Series) -> VolatilityResult:
        if self._result is None:
            raise RuntimeError("Call fit() first")
        res = self._result
        # Conditional vol is in % units — convert back and annualise
        cond_vol = (res.conditional_volatility / 100) * np.sqrt(252)
        cond_vol.name = "conditional_vol"

        # One-step-ahead forecast
        fc = res.forecast(horizon=1, reindex=False)
        fc_vol = float(np.sqrt(fc.variance.values[-1, 0]) / 100) * np.sqrt(252)
        forecast_vol = pd.Series([fc_vol], index=[log_returns.dropna().index[-1] + pd.Timedelta("1D")])

        return VolatilityResult(
            symbol="arabica",
            conditional_vol=cond_vol,
            forecast_vol=forecast_vol,
            model_name=self.model_name,
        )

    def fit_predict(self, log_returns: pd.Series) -> VolatilityResult:
        return self.fit(log_returns).get_result(log_returns)
