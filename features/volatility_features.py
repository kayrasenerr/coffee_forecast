"""
features/volatility_features.py
Volatility-based features: realised vol, Parkinson, vol-of-vol, vol ratio.

These feed:
  1. GARCH model as diagnostics
  2. HMM as regime-detection inputs
  3. SARIMAX as volatility regime controls
"""
import numpy as np
import pandas as pd
from config.settings import FeatureConfig
from features.base import FeatureBuilder
from schemas.types import PriceFrame


class VolatilityFeatureBuilder(FeatureBuilder):
    name = "vol"

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build(self, frames: dict[str, PriceFrame]) -> pd.DataFrame:
        close = self._safe_close(frames, "arabica")
        if close is None:
            raise ValueError("Arabica price data required")

        pf = frames["arabica"].data
        log_ret = np.log(close).diff()
        w = self.cfg.vol_window

        feats: dict[str, pd.Series] = {}

        # ── Realised close-to-close vol (annualised) ──────────────────
        feats["realised_vol"] = log_ret.rolling(w).std() * np.sqrt(252)

        # ── Parkinson high-low vol estimator (more efficient) ─────────
        if {"high", "low"}.issubset(pf.columns):
            hl = np.log(pf["high"] / pf["low"]) ** 2
            feats["parkinson_vol"] = (
                (1 / (4 * np.log(2))) * hl.rolling(w).mean()
            ).apply(lambda x: np.sqrt(x * 252))

        # ── Volatility ratio (short/long) — clustering indicator ──────
        short_vol = log_ret.rolling(5).std() * np.sqrt(252)
        long_vol  = log_ret.rolling(63).std() * np.sqrt(252)
        feats["vol_ratio"] = (short_vol / long_vol.replace(0, np.nan))

        # ── Vol-of-vol (second-order uncertainty) ─────────────────────
        feats["vol_of_vol"] = feats["realised_vol"].rolling(w).std()

        # ── Absolute return (GARCH input proxy) ───────────────────────
        feats["abs_ret"] = log_ret.abs()
        feats["sq_ret"]  = log_ret ** 2

        df = pd.DataFrame(feats).sort_index()
        return df.add_prefix(f"{self.name}_") if self.name else df
