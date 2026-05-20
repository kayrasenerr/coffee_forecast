"""
features/pipeline.py
Assembles all FeatureBuilders into a single aligned FeatureMatrix.

Usage:
    pipeline = FeaturePipeline(settings.features)
    fm = pipeline.build(frames)
"""
import pandas as pd
import numpy as np
from config.settings import FeatureConfig
from features.price_features import PriceFeatureBuilder
from features.volatility_features import VolatilityFeatureBuilder
from schemas.types import PriceFrame, FeatureMatrix


class FeaturePipeline:
    def __init__(self, cfg: FeatureConfig):
        self.builders = [
            PriceFeatureBuilder(cfg),
            VolatilityFeatureBuilder(cfg),
        ]

    def build(self, frames: dict[str, PriceFrame], symbol: str = "arabica") -> FeatureMatrix:
        """
        Runs all builders, joins on date index, drops NaN rows.
        Returns FeatureMatrix with target = next-day log return.
        """
        parts: list[pd.DataFrame] = []
        for builder in self.builders:
            try:
                df = builder.build(frames)
                parts.append(df)
            except ValueError as e:
                print(f"[WARN] {builder.name}: {e}")

        if not parts:
            raise RuntimeError("No features built — check data availability")

        combined = pd.concat(parts, axis=1).sort_index()

        # Target: next-day log return (shift -1)
        log_ret_col = "price_log_ret"
        if log_ret_col not in combined.columns:
            raise RuntimeError(f"Expected column '{log_ret_col}' not found")

        target = combined[log_ret_col].shift(-1).rename("target")

        # Drop lookahead and last row (no target)
        combined = combined.join(target)
        combined = combined.dropna(subset=["target"])
        combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.7))  # drop sparse cols
        combined = combined.ffill().dropna()

        feature_cols = [c for c in combined.columns if c != "target"]
        return FeatureMatrix(
            features=combined[feature_cols],
            target=combined["target"],
            symbol=symbol,
        )

    def get_hmm_inputs(self, fm: FeatureMatrix) -> np.ndarray:
        """
        Compact 3-column observation matrix for HMM:
        [log_ret, realised_vol, fx_zscore]
        Falls back gracefully if FX unavailable.
        """
        cols = ["price_log_ret", "vol_realised_vol"]
        fx_col = "price_fx_usd_brl_zscore"
        if fx_col in fm.features.columns:
            cols.append(fx_col)
        return fm.features[cols].dropna().values
