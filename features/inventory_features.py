"""
features/inventory_features.py
================================
Inventory / certified-stock feature engineering.

Key signals:
  - ICE certified stock levels (absolute + trend)
  - Stock depletion velocity (rate of drawdown)
  - Days-of-consumption coverage ratio
  - Pending-grading stock (leading indicator of certified levels)
  - YoY stock comparison

Market interpretation:
  Low certified stocks  → supply tightness → bullish price pressure
  Rapid drawdown        → demand > supply  → escalating stress
  Stock build           → oversupply       → bearish pressure
  Stocks < 1M bags      → historically associated with price spikes

Data sources (when available):
  - ICE Exchange certified stock reports (daily)
  - ICO carry-out stocks (monthly, from PSD)
  - USDA ending stocks (annual/marketing-year)
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
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[-1] > x[:-1]).mean() * 100, raw=True
    )


class InventoryFeatureTransformer(FeatureTransformerBase):
    """
    Compute inventory / stock features from certified stock data.

    Required inputs:
      "inventory" : DataFrame with columns:
          certified_lots   (ICE lots; 1 lot = 250 bags = 37,500 lbs)
          [pending_lots]   optional

    Optional inputs:
      "usda_stocks" : DataFrame with country-level ending_stocks columns
    """

    feature_group = "inventory"

    def __init__(
        self,
        depletion_window: int = 21,      # days for velocity calculation
        stress_threshold_lots: float = 3000,  # below = supply stress zone
    ):
        self.depletion_window = depletion_window
        self.stress_threshold_lots = stress_threshold_lots

    def get_required_inputs(self) -> List[str]:
        return ["inventory"]

    @property
    def output_columns(self) -> List[str]:
        return [
            "certified_lots",
            "certified_lots_z_252d",
            "certified_lots_pctile_252d",
            "certified_lots_yoy_pct",
            "stock_depletion_velocity",
            "stock_stress_flag",
            "stock_coverage_days",
            "pending_certified_ratio",
        ]

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        inv = inputs["inventory"].copy()
        f = pd.DataFrame(index=inv.index)

        stocks = inv["certified_lots"]

        # ----------------------------------------------------------------
        # Level and anomaly
        # ----------------------------------------------------------------
        f["certified_lots"] = stocks
        f["certified_lots_z_252d"] = _zscore(stocks, 252)
        f["certified_lots_pctile_252d"] = _percentile_rank(stocks, 252)

        # YoY change
        trading_days_year = 252
        f["certified_lots_yoy_pct"] = (
            (stocks - stocks.shift(trading_days_year)) / stocks.shift(trading_days_year) * 100
        )

        # ----------------------------------------------------------------
        # Depletion velocity: rate of stock change (lots/day)
        # ----------------------------------------------------------------
        f["stock_depletion_velocity"] = -stocks.diff(self.depletion_window) / self.depletion_window
        # Positive = stock being drawn down (bullish)
        # Negative = stock being built (bearish)
        f["stock_depletion_z_63d"] = _zscore(f["stock_depletion_velocity"], 63)

        # ----------------------------------------------------------------
        # Stress flag
        # ----------------------------------------------------------------
        f["stock_stress_flag"] = (stocks < self.stress_threshold_lots).astype(float)
        f["stock_critically_low"] = (stocks < self.stress_threshold_lots * 0.5).astype(float)

        # ----------------------------------------------------------------
        # Stock coverage days (estimate)
        # Based on typical annual consumption ~170M bags for arabica
        # ----------------------------------------------------------------
        annual_consumption_lots = 170e6 / 250 / 252  # lots per day world estimate
        f["stock_coverage_days"] = stocks / annual_consumption_lots

        # ----------------------------------------------------------------
        # Pending / certified ratio (leading indicator)
        # ----------------------------------------------------------------
        if "pending_lots" in inv.columns:
            pending = inv["pending_lots"]
            denom = (stocks + pending).replace(0, np.nan)
            f["pending_certified_ratio"] = pending / denom
        else:
            f["pending_certified_ratio"] = np.nan

        # ----------------------------------------------------------------
        # USDA country-level ending stocks (if available)
        # ----------------------------------------------------------------
        if "usda_stocks" in inputs:
            usda = inputs["usda_stocks"]
            for col in usda.columns:
                if "ending_stocks" in col:
                    country = col.replace("_ending_stocks", "")
                    series = usda[col]
                    f[f"usda_stocks_{country}"] = series
                    f[f"usda_stocks_{country}_yoy"] = series.pct_change(1) * 100

        return f.dropna(how="all")
