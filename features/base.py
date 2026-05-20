"""
features/base.py
Abstract contract for all feature builders.
Each builder receives raw PriceFrames, emits a named pd.DataFrame column set.
"""
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from schemas.types import PriceFrame


class FeatureBuilder(ABC):
    name: str = ""          # used as column prefix

    @abstractmethod
    def build(self, frames: dict[str, PriceFrame]) -> pd.DataFrame:
        """
        Args:
            frames: dict of canonical PriceFrames keyed by logical name
        Returns:
            DataFrame with DatetimeIndex; columns prefixed with self.name
        """
        ...

    def _safe_close(self, frames: dict[str, PriceFrame], key: str) -> Optional[pd.Series]:
        pf = frames.get(key)
        if pf is None or pf.data.empty:
            return None
        return pf.close()
