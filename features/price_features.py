"""
features/price_features.py
===========================
Price-derived feature transformations.

Feature groups computed:
  1. Log returns at multiple horizons
  2. Rolling volatility (realised vol)
  3. Momentum signals
  4. Arabica-Robusta spread dynamics
  5. Futures curve features (when multi-contract data available)
  6. Seasonal decomposition residuals

Philosophy:
  - Anomaly-based where possible (z-score over rolling window)
  - Lagged to avoid lookahead
  - Named clearly for interpretability
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from contracts.interfaces import FeatureTransformerBase
from contracts.schemas import CoffeeVariety, DataFrequency

logger = logging.getLogger(__name__)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _log_return(prices: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(prices / prices.shift(lag))


def _realised_vol(log_returns: pd.Series, window: int, annualise: int = 252) -> pd.Series:
    """Annualised realised volatility."""
    return log_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(annualise)


class PriceFeatureTransformer(FeatureTransformerBase):
    """
    Compute price-based features from futures OHLCV data.

    Required inputs key: "prices"  (DataFrame with 'close' column)
    Optional inputs key: "prices_robusta" (for spread features)
    """

    feature_group = "price"

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        include_spread: bool = True,
        include_momentum: bool = True,
        include_volatility: bool = True,
        annualise_days: int = 252,
    ):
        self.windows = windows or [5, 10, 21, 63]
        self.include_spread = include_spread
        self.include_momentum = include_momentum
        self.include_volatility = include_volatility
        self.annualise_days = annualise_days

    def get_required_inputs(self) -> List[str]:
        return ["prices"]

    @property
    def output_columns(self) -> List[str]:
        cols = []
        for w in self.windows:
            cols += [
                f"log_return_{w}d",
                f"realised_vol_{w}d",
                f"momentum_z_{w}d",
                f"price_z_{w}d",
            ]
        if self.include_spread:
            cols += ["arabica_robusta_spread", "spread_z_21d", "spread_percentile_252d"]
        return cols

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        prices_df = inputs["prices"]
        close = prices_df["close"].copy()

        features = pd.DataFrame(index=prices_df.index)

        # ----------------------------------------------------------------
        # 1. Log returns at each window
        # ----------------------------------------------------------------
        log_ret_1d = _log_return(close, lag=1)
        features["log_return_1d"] = log_ret_1d

        for w in self.windows:
            features[f"log_return_{w}d"] = _log_return(close, lag=w)

        # ----------------------------------------------------------------
        # 2. Realised volatility
        # ----------------------------------------------------------------
        if self.include_volatility:
            for w in self.windows:
                features[f"realised_vol_{w}d"] = _realised_vol(
                    log_ret_1d, w, self.annualise_days
                )
            # Vol-of-vol (vol regime signal)
            vol_21 = features["realised_vol_21d"] if "realised_vol_21d" in features else None
            if vol_21 is not None:
                features["vol_of_vol_21d"] = vol_21.rolling(21, min_periods=10).std()
                features["vol_z_63d"] = _zscore(vol_21, 63)

        # ----------------------------------------------------------------
        # 3. Price level anomaly (z-score vs rolling mean)
        # ----------------------------------------------------------------
        for w in self.windows:
            features[f"price_z_{w}d"] = _zscore(close, w)

        # ----------------------------------------------------------------
        # 4. Momentum (sign-normalised return)
        # ----------------------------------------------------------------
        if self.include_momentum:
            for w in self.windows:
                ret = features[f"log_return_{w}d"]
                features[f"momentum_z_{w}d"] = _zscore(ret, w * 3)

            # RSI-style overbought/oversold indicator
            features["rsi_14"] = self._rsi(close, 14)

            # 52-week high/low proximity
            roll_hi = close.rolling(252, min_periods=126).max()
            roll_lo = close.rolling(252, min_periods=126).min()
            features["pct_from_52w_high"] = (close - roll_hi) / roll_hi
            features["pct_from_52w_low"] = (close - roll_lo) / (roll_lo + 1e-8)

        # ----------------------------------------------------------------
        # 5. Arabica-Robusta spread (cross-variety signal)
        # ----------------------------------------------------------------
        if self.include_spread and "prices_robusta" in inputs:
            rob_close = inputs["prices_robusta"]["close"]
            # Normalise: Arabica is cents/lb, Robusta is USD/tonne
            # 1 USD/tonne = 0.04536 cents/lb → convert robusta to cents/lb
            rob_close_norm = rob_close * 0.04536
            spread = close - rob_close_norm
            features["arabica_robusta_spread"] = spread
            features["spread_z_21d"] = _zscore(spread, 21)
            features["spread_z_63d"] = _zscore(spread, 63)
            features["spread_percentile_252d"] = spread.rolling(252, min_periods=126).apply(
                lambda x: (x[-1] > x).mean() * 100, raw=True
            )

        # ----------------------------------------------------------------
        # 6. OHLC-based microstructure features
        # ----------------------------------------------------------------
        if all(c in prices_df.columns for c in ["high", "low", "open", "close"]):
            features["daily_range"] = (prices_df["high"] - prices_df["low"]) / prices_df["close"]
            features["daily_range_z_21d"] = _zscore(features["daily_range"], 21)
            # Gap: close-to-open
            features["overnight_gap"] = np.log(prices_df["open"] / prices_df["close"].shift(1))

        return features.dropna(how="all")

    @staticmethod
    def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class CurveFeatureTransformer(FeatureTransformerBase):
    """
    Futures curve / term-structure features.

    Key signals:
      - Front-back spread (M1 - M2): backwardation vs contango
      - Carry: annualised roll yield
      - Curve slope z-score

    Contango (M1 < M2) → ample supply, storage abundant.
    Backwardation (M1 > M2) → supply stress, immediate demand.
    """

    feature_group = "curve"

    def get_required_inputs(self) -> List[str]:
        return ["curve"]

    @property
    def output_columns(self) -> List[str]:
        return [
            "curve_front_back_spread",
            "curve_spread_z_63d",
            "curve_carry_annualised",
            "is_backwardation",
        ]

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        curve_df = inputs["curve"]
        features = pd.DataFrame(index=curve_df.index)

        m1_col = next((c for c in curve_df.columns if "m1" in c.lower()), None)
        m2_col = next((c for c in curve_df.columns if "m2" in c.lower()), None)

        if m1_col and m2_col:
            spread = curve_df[m1_col] - curve_df[m2_col]
            features["curve_front_back_spread"] = spread
            features["curve_spread_z_63d"] = _zscore(spread, 63)
            # Approximate annualised carry (assuming ~2mo between contracts)
            features["curve_carry_annualised"] = (-spread / curve_df[m1_col]) * 6
            features["is_backwardation"] = (spread > 0).astype(float)

        return features
