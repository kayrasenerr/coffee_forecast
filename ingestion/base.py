"""
ingestion/base.py
=================
Shared utilities for all data source adapters.
Concrete sources inherit from DataSourceBase (contracts/interfaces.py).
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Callable, Optional

import pandas as pd

from contracts.interfaces import DataSourceBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry decorator for flaky external APIs
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Simple exponential-backoff retry decorator."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay_seconds
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "Attempt %d/%d failed (%s). Retrying in %.1fs …",
                        attempt, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Common DataFrame normalisation helpers
# ---------------------------------------------------------------------------

def normalise_index(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has a UTC-normalised DatetimeIndex.
    Optionally resample to a target frequency.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)           # strip tz for simplicity
    df.index.name = "date"
    df = df.sort_index()
    if freq:
        df = df.resample(freq).last()
    return df


def forward_fill_gaps(df: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Forward-fill gaps up to max_gap_days (for weekends / holidays)."""
    return df.ffill(limit=max_gap_days)


def drop_leading_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ALL columns are NaN (typical at start of series)."""
    return df.dropna(how="all")


# ---------------------------------------------------------------------------
# Base class with shared logic
# ---------------------------------------------------------------------------

class CachingDataSource(DataSourceBase):
    """
    DataSourceBase that adds optional local parquet caching.

    Subclasses implement _fetch_remote() instead of fetch().
    """

    source_id: str = "base"

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir

    def _cache_path(self, start: date, end: date) -> Optional[Any]:
        if self._cache_dir is None:
            return None
        from pathlib import Path
        p = Path(self._cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        fname = f"{self.source_id}_{start}_{end}.parquet"
        return p / fname

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError

    def fetch(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        cache_path = self._cache_path(start, end)
        if cache_path and cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return pd.read_parquet(cache_path)

        logger.info("[%s] Fetching %s → %s", self.source_id, start, end)
        df = self._fetch_remote(start, end, **kwargs)

        if cache_path:
            df.to_parquet(cache_path)
            logger.debug("Cached to %s", cache_path)

        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default: just normalise index and drop all-NaN rows."""
        df = normalise_index(df)
        df = drop_leading_nans(df)
        return df
