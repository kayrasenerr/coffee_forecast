"""
ingestion/fx.py
===============
FX rate ingestion for coffee-relevant currency pairs.

Primary source: FRED (Federal Reserve Economic Data).
Requires: FRED API key set as COFFEE_FRED_API_KEY env var.

Key series:
  DEXBZUS  →  USD/BRL (Brazil)
  DEXVNUS  →  USD/VND (Vietnam)
  DEXCOUS  →  USD/COP (Colombia)
  DEXUSEU  →  EUR/USD (European demand)
  DEXUSNZ  →  USD/UGX proxy (Uganda — use YF fallback)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from ingestion.base import CachingDataSource, forward_fill_gaps, normalise_index

logger = logging.getLogger(__name__)


class FredFXSource(CachingDataSource):
    """
    Fetch daily FX spot rate from FRED.

    Falls back to Yahoo Finance if FRED key is unavailable.
    """

    def __init__(
        self,
        series_id: str,
        pair: str,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.series_id = series_id
        self.pair = pair
        self.source_id = f"fred_fx_{pair.lower()}"

    def _fetch_via_fred(self, start: date, end: date) -> pd.DataFrame:
        import os

        api_key = os.environ.get("COFFEE_FRED_API_KEY") or os.environ.get("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError("COFFEE_FRED_API_KEY not set")

        try:
            from fredapi import Fred
        except ImportError as e:
            raise ImportError("pip install fredapi") from e

        fred = Fred(api_key=api_key)
        series = fred.get_series(
            self.series_id,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        df = series.to_frame(name="rate")
        df["pair"] = self.pair
        return df

    def _fetch_via_yahoo(self, start: date, end: date) -> pd.DataFrame:
        """Fallback: Yahoo Finance FX."""
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("pip install yfinance") from e

        # Convert FRED pair to Yahoo format
        yahoo_map = {
            "USDBRL": "USDBRL=X",
            "USDVND": "USDVND=X",
            "USDCOP": "USDCOP=X",
            "EURUSD": "EURUSD=X",
            "USDUGX": "USDUGX=X",
        }
        ticker = yahoo_map.get(self.pair, f"{self.pair}=X")
        raw = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
        )
        if raw.empty:
            return pd.DataFrame()
        df = raw[["Close"]].rename(columns={"Close": "rate"})
        df["pair"] = self.pair
        return df

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            df = self._fetch_via_fred(start, end)
            logger.debug("[%s] Fetched via FRED", self.source_id)
            return df
        except (EnvironmentError, Exception) as exc:
            logger.warning("[%s] FRED unavailable (%s), trying Yahoo …", self.source_id, exc)
            return self._fetch_via_yahoo(start, end)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        df = df[df["rate"] > 0]
        df = forward_fill_gaps(df, max_gap_days=5)
        return df
