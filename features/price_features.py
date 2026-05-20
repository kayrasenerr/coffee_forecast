"""
features/price_features.py
Causal and momentum features derived from price and FX data.

Design philosophy:
  - Anomaly-based (z-scores) over raw levels.
  - Lagged features to avoid lookahead bias.
  - FX as a causal exogenous variable, not just correlation.
"""
import numpy as np
import pandas as pd
from config.settings import FeatureConfig
from features.base import FeatureBuilder
from schemas.types import PriceFrame


class PriceFeatureBuilder(FeatureBuilder):
    name = "price"

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build(self, frames: dict[str, PriceFrame]) -> pd.DataFrame:
        close = self._safe_close(frames, "arabica")
        if close is None:
            raise ValueError("Arabica price data required")

        feats: dict[str, pd.Series] = {}
        log_ret = np.log(close).diff()

        # ── Momentum ──────────────────────────────────────────────────
        for w in self.cfg.momentum_windows:
            feats[f"mom_{w}d"] = np.log(close / close.shift(w))

        # ── Z-score of returns (anomaly signal) ───────────────────────
        feats["zscore_ret"] = self._zscore(log_ret, self.cfg.zscore_window)

        # ── Z-score of price relative to rolling mean ─────────────────
        feats["zscore_price"] = self._zscore(close, self.cfg.zscore_window)

        # ── Log return (primary target & model input) ─────────────────
        feats["log_ret"] = log_ret
        feats["log_ret_lag1"] = log_ret.shift(1)
        feats["log_ret_lag5"] = log_ret.shift(5)

        # ── Realised direction (for target construction) ───────────────
        feats["direction"] = np.sign(log_ret)

        # ── FX: USD/BRL as causal variable ───────────────────────────
        usd_brl = self._safe_close(frames, "usd_brl")
        if usd_brl is not None:
            usd_brl = usd_brl.reindex(close.index).ffill()
            fx_ret = np.log(usd_brl).diff()
            feats["fx_usd_brl"]        = usd_brl
            feats["fx_usd_brl_ret"]    = fx_ret
            feats["fx_usd_brl_zscore"] = self._zscore(usd_brl, self.cfg.zscore_window)
            # Lag FX by 1 day → cleaner causal signal
            feats["fx_usd_brl_ret_lag1"] = fx_ret.shift(1)

        df = pd.DataFrame(feats).sort_index()
        return df.add_prefix(f"{self.name}_") if self.name else df

    @staticmethod
    def _zscore(s: pd.Series, window: int) -> pd.Series:
        mu = s.rolling(window, min_periods=window // 2).mean()
        sd = s.rolling(window, min_periods=window // 2).std()
        return (s - mu) / sd.replace(0, np.nan)
