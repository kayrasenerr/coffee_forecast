"""
features/positioning_features.py
=================================
CFTC COT (Commitments of Traders) feature engineering.

Key signals:
  - Net non-commercial (speculative) position → sentiment
  - Commercial hedger position → producer hedging pressure
  - Extreme positioning flags → mean-reversion signal
  - Position changes (delta) → momentum / capitulation

Academic basis:
  - Large non-commercial net long → crowded trade risk
  - Extreme commercial short → high producer hedging → potential bear trap
  - Position unwinding → price cascade risk
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
    """Rolling percentile rank (0–100) over `window` periods."""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda x: (x[-1] > x[:-1]).mean() * 100, raw=True
    )


class COTFeatureTransformer(FeatureTransformerBase):
    """
    Compute positioning features from CFTC COT report data.

    Required inputs:
      "cot" : DataFrame with columns:
          noncommercial_long, noncommercial_short,
          commercial_long, commercial_short,
          open_interest  (optional)
    """

    feature_group = "positioning"

    def __init__(
        self,
        normalise_by_oi: bool = True,
        extreme_window: int = 52,       # weeks
        extreme_threshold: float = 90,  # percentile
    ):
        self.normalise_by_oi = normalise_by_oi
        self.extreme_window = extreme_window
        self.extreme_threshold = extreme_threshold

    def get_required_inputs(self) -> List[str]:
        return ["cot"]

    @property
    def output_columns(self) -> List[str]:
        return [
            "cot_net_noncomm",
            "cot_net_noncomm_norm",
            "cot_net_comm",
            "cot_net_noncomm_z_52w",
            "cot_net_noncomm_pctile_52w",
            "cot_net_noncomm_delta_4w",
            "cot_extreme_long",
            "cot_extreme_short",
            "cot_crowded_long",
        ]

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        cot = inputs["cot"].copy()

        f = pd.DataFrame(index=cot.index)

        # ----------------------------------------------------------------
        # Net positions
        # ----------------------------------------------------------------
        f["cot_net_noncomm"] = (
            cot["noncommercial_long"] - cot["noncommercial_short"]
        )
        f["cot_net_comm"] = (
            cot["commercial_long"] - cot["commercial_short"]
        )
        f["cot_total_oi"] = cot.get("open_interest", pd.Series(np.nan, index=cot.index))

        # ----------------------------------------------------------------
        # Normalise by open interest
        # ----------------------------------------------------------------
        if self.normalise_by_oi and "open_interest" in cot.columns:
            oi = cot["open_interest"].replace(0, np.nan)
            f["cot_net_noncomm_norm"] = f["cot_net_noncomm"] / oi * 100  # %
            f["cot_net_comm_norm"] = f["cot_net_comm"] / oi * 100
            f["cot_long_ratio"] = cot["noncommercial_long"] / oi * 100
            f["cot_short_ratio"] = cot["noncommercial_short"] / oi * 100
        else:
            f["cot_net_noncomm_norm"] = f["cot_net_noncomm"]

        # ----------------------------------------------------------------
        # Z-score and percentile rank (anomaly detection)
        # ----------------------------------------------------------------
        net = f["cot_net_noncomm_norm"]
        f["cot_net_noncomm_z_52w"] = _zscore(net, self.extreme_window)
        f["cot_net_noncomm_pctile_52w"] = _percentile_rank(net, self.extreme_window)

        # ----------------------------------------------------------------
        # Change in positioning (momentum / capitulation signal)
        # ----------------------------------------------------------------
        f["cot_net_noncomm_delta_4w"] = f["cot_net_noncomm_norm"].diff(4)
        f["cot_net_noncomm_delta_13w"] = f["cot_net_noncomm_norm"].diff(13)

        # ----------------------------------------------------------------
        # Extreme positioning flags
        # ----------------------------------------------------------------
        pct = f["cot_net_noncomm_pctile_52w"]
        f["cot_extreme_long"] = (pct >= self.extreme_threshold).astype(float)
        f["cot_extreme_short"] = (pct <= (100 - self.extreme_threshold)).astype(float)
        f["cot_crowded_long"] = (f["cot_net_noncomm_z_52w"] >= 2.0).astype(float)

        # ----------------------------------------------------------------
        # Resample to target frequency (COT is weekly; upsample to daily)
        # ----------------------------------------------------------------
        f = f.resample(frequency.value).last().ffill(limit=7)

        return f
