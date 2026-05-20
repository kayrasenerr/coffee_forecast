"""
features/macro_features.py
===========================
Macro-economic feature engineering for coffee pricing.

Key causal relationships:
  USD/BRL appreciation → Brazilian farmers receive more BRL/bag
                       → incentive to sell → supply increase → price bearish
  USD/BRL depreciation → farmers hold back → supply squeeze → price bullish
  EUR/USD → European roaster purchasing power / demand proxy
  USD/VND → Vietnamese Robusta competitiveness

This is one of the most direct, well-documented causal pathways
in coffee markets. BRL is THE single most important FX driver
for Arabica coffee.
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


def _log_return(s: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(s / s.shift(lag))


class MacroFeatureTransformer(FeatureTransformerBase):
    """
    Compute macro-economic features from FX and economic data.

    Required inputs:
      "fx_usdbrl"   : DataFrame with 'rate' column
      "fx_eurusd"   : DataFrame with 'rate' column
    Optional:
      "fx_usdvnd"   : Robusta-relevant
      "fx_usdcop"   : Colombian Arabica
    """

    feature_group = "macro"

    def __init__(
        self,
        fx_windows: Optional[List[int]] = None,
        include_brl_carry: bool = True,
    ):
        self.fx_windows = fx_windows or [5, 21, 63]
        self.include_brl_carry = include_brl_carry

    def get_required_inputs(self) -> List[str]:
        return ["fx_usdbrl"]

    @property
    def output_columns(self) -> List[str]:
        cols = []
        for pair in ["usdbrl", "eurusd", "usdvnd"]:
            for w in self.fx_windows:
                cols += [f"{pair}_return_{w}d", f"{pair}_z_{w}d"]
        cols += ["brl_strength_index", "brl_farmer_incentive"]
        return cols

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        f = pd.DataFrame()

        # ----------------------------------------------------------------
        # USD/BRL — primary driver for Arabica
        # ----------------------------------------------------------------
        if "fx_usdbrl" in inputs:
            brl = inputs["fx_usdbrl"]["rate"]
            f = self._add_fx_features(f, brl, "usdbrl")

            # BRL Farmer incentive proxy:
            # Higher USDBRL = more BRL per USD bag → sell pressure
            # We invert so positive = supply-bearish
            brl_30d_ma = brl.rolling(30, min_periods=15).mean()
            f["brl_strength_index"] = (brl / brl_30d_ma - 1) * 100  # % above MA
            f["brl_farmer_incentive"] = _zscore(brl, 252)  # high z → sell pressure
            f["brl_above_200d_ma"] = (brl > brl.rolling(200, min_periods=100).mean()).astype(float)

        # ----------------------------------------------------------------
        # EUR/USD — European demand proxy
        # ----------------------------------------------------------------
        if "fx_eurusd" in inputs:
            eur = inputs["fx_eurusd"]["rate"]
            f = self._add_fx_features(f, eur, "eurusd")
            # Strong EUR → European roasters buy more → demand bullish
            f["eur_demand_proxy"] = _zscore(eur, 63)

        # ----------------------------------------------------------------
        # USD/VND — Vietnamese Robusta competitiveness
        # ----------------------------------------------------------------
        if "fx_usdvnd" in inputs:
            vnd = inputs["fx_usdvnd"]["rate"]
            f = self._add_fx_features(f, vnd, "usdvnd")
            # Weak VND → Vietnamese exports cheaper → Robusta bearish
            f["vnd_competitive"] = _zscore(vnd, 63)

        # ----------------------------------------------------------------
        # USD/COP — Colombian Arabica competitiveness
        # ----------------------------------------------------------------
        if "fx_usdcop" in inputs:
            cop = inputs["fx_usdcop"]["rate"]
            f = self._add_fx_features(f, cop, "usdcop")

        # ----------------------------------------------------------------
        # Dollar Index proxy (equal-weight of BRL, EUR, VND)
        # ----------------------------------------------------------------
        brl_z = f.get("usdbrl_z_21d")
        eur_z = f.get("eurusd_z_21d")
        if brl_z is not None and eur_z is not None:
            f["coffee_fx_index"] = (brl_z - eur_z) / 2  # USD strength composite

        return f.dropna(how="all")

    def _add_fx_features(
        self, f: pd.DataFrame, rate: pd.Series, prefix: str
    ) -> pd.DataFrame:
        """Add return and z-score features for an FX rate series."""
        f = f.copy()
        log_ret_1d = _log_return(rate)

        for w in self.fx_windows:
            f[f"{prefix}_return_{w}d"] = _log_return(rate, lag=w)
            f[f"{prefix}_z_{w}d"] = _zscore(rate, w)

        # Realised FX volatility
        f[f"{prefix}_vol_21d"] = log_ret_1d.rolling(21, min_periods=10).std() * np.sqrt(252)

        return f
