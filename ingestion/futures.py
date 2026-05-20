"""
ingestion/futures.py
====================
Coffee futures price ingestion.

Primary: Yahoo Finance (yfinance) — free, no key required.
Columns returned: open, high, low, close, volume, adj_close

Arabica : KC=F  (ICE New York, cents/lb)
Robusta : RC=F  (ICE London, USD/tonne)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from contracts.schemas import CoffeeVariety, Exchange
from ingestion.base import CachingDataSource, forward_fill_gaps, normalise_index

logger = logging.getLogger(__name__)


class YahooFuturesSource(CachingDataSource):
    """
    Fetch coffee futures OHLCV from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker   : Yahoo Finance ticker symbol (e.g. "KC=F")
    variety  : CoffeeVariety.ARABICA or ROBUSTA
    exchange : Exchange enum value
    cache_dir: optional local parquet cache directory
    """

    def __init__(
        self,
        ticker: str,
        variety: CoffeeVariety,
        exchange: Exchange,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.ticker = ticker
        self.variety = variety
        self.exchange = exchange
        self.source_id = f"yahoo_{variety.value}_futures"

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("pip install yfinance") from e

        ticker_obj = yf.Ticker(self.ticker)
        df = ticker_obj.history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
        )
        if df.empty:
            logger.warning("[%s] Empty response for %s → %s", self.source_id, start, end)
            return pd.DataFrame()

        df = df.rename(columns=str.lower)
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep_cols].copy()
        df["variety"] = self.variety.value
        df["exchange"] = self.exchange.value
        df["symbol"] = self.ticker
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        # Remove zero or negative prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df = df[df[col] > 0]
        df = forward_fill_gaps(df, max_gap_days=5)
        return df


class MultiContractFuturesSource(CachingDataSource):
    """
    Fetch multiple futures contracts (M1, M2, M3) for curve construction.

    Note: Free sources only provide continuous front-month.
    This stub is ready for ICE direct feed integration.
    """

    def __init__(self, variety: CoffeeVariety, cache_dir: Optional[str] = None):
        super().__init__(cache_dir=cache_dir)
        self.variety = variety
        self.source_id = f"multi_contract_{variety.value}"

        # Ticker map: tenor label → Yahoo symbol
        # These are continuous contracts — use for spread/curve approximation
        if variety == CoffeeVariety.ARABICA:
            self.tickers: dict[str, str] = {
                "M1": "KC=F",
                # M2, M3 require paid data feed
            }
        else:
            self.tickers = {
                "M1": "RC=F",
            }

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("pip install yfinance") from e

        frames: list[pd.DataFrame] = []
        for tenor, ticker in self.tickers.items():
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,
            )
            if raw.empty:
                continue
            close = raw["Close"].rename(f"close_{tenor}")
            frames.append(close)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1)
        df["variety"] = self.variety.value
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        return normalise_index(df)
