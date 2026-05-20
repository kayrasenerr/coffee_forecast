"""
ingestion/base.py
Abstract contract every data source must satisfy.
Add new sources (ICO, USDA, NOAA) by subclassing DataSource.
"""
from abc import ABC, abstractmethod
from datetime import date
import pandas as pd
from schemas.types import PriceFrame


class DataSource(ABC):
    """
    Minimal interface for any ingestion module.
    Downstream code depends only on this contract.
    """
    source_id: str = ""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: date,
        end: date,
        **kwargs,
    ) -> PriceFrame:
        """Fetch raw data and return a canonical PriceFrame."""
        ...

    def _validate_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure expected columns exist; lowercase col names."""
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.source_id}: missing columns {missing}")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.dropna(subset=["close"])
