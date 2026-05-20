"""
schemas/types.py
Canonical data contracts shared across all modules.
Every module speaks this language — no raw dict passing.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PriceFrame:
    """Standardised OHLCV block returned by every ingestion source."""
    symbol: str
    data: pd.DataFrame          # index=DatetimeIndex, cols=[open,high,low,close,volume]
    source: str = ""
    currency: str = "USD"

    def close(self) -> pd.Series:
        return self.data["close"]

    def log_returns(self) -> pd.Series:
        return np.log(self.close()).diff().rename(f"{self.symbol}_log_ret")


@dataclass
class FeatureMatrix:
    """Aligned feature table fed into all models."""
    features: pd.DataFrame      # index=DatetimeIndex, cols=feature names
    target: pd.Series           # what we're forecasting (e.g. next-day log return)
    symbol: str = ""

    def train_test_split(self, test_start: pd.Timestamp) -> tuple["FeatureMatrix", "FeatureMatrix"]:
        mask = self.features.index < test_start
        train = FeatureMatrix(self.features[mask], self.target[mask], self.symbol)
        test  = FeatureMatrix(self.features[~mask], self.target[~mask], self.symbol)
        return train, test


@dataclass
class RegimeResult:
    """Output of any regime-detection model."""
    symbol: str
    states: pd.Series           # index=DatetimeIndex, values=int regime labels
    state_probs: pd.DataFrame   # index=DatetimeIndex, cols=regime probs
    state_labels: dict          # {0: "low_stress", 1: "mid_stress", 2: "high_stress"}
    model_name: str = "HMM"


@dataclass
class ForecastResult:
    """Output of any point/probabilistic forecast model."""
    symbol: str
    forecast_dates: pd.DatetimeIndex
    mean: pd.Series
    lower: pd.Series            # e.g. 10th percentile
    upper: pd.Series            # e.g. 90th percentile
    model_name: str = ""
    horizon: int = 1
    metric: Optional[dict] = field(default=None)


@dataclass
class VolatilityResult:
    """Output of GARCH-family volatility models."""
    symbol: str
    conditional_vol: pd.Series  # index=DatetimeIndex
    forecast_vol: pd.Series     # next-step forecasted vol
    model_name: str = "GARCH"


@dataclass
class BacktestReport:
    """Walk-forward validation summary."""
    symbol: str
    model_name: str
    predictions: pd.DataFrame   # cols=[date, actual, forecast, regime]
    directional_accuracy: float
    rmse: float
    hit_rate_by_regime: dict    # {regime_label: accuracy}
    n_folds: int
