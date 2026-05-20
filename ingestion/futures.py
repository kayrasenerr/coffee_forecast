"""
ingestion/futures.py
Live coffee futures prices via Yahoo Finance.

  KC=F  — ICE Arabica (cents/lb, continuous front month)
  RB=F  — LIFFE Robusta ($/MT) — may be absent on yfinance

In sandboxed environments where finance.yahoo.com is blocked, yfinance
falls back to internally generated synthetic data automatically (labelled
"Mode=synthetic" in its output). The schema contract is identical either way.

For production, ensure outbound HTTPS to:
  query1.finance.yahoo.com
  query2.finance.yahoo.com
  finance.yahoo.com
"""
import yfinance as yf
import pandas as pd
from datetime import date
from ingestion.base import DataSource
from schemas.types import PriceFrame


class YFinanceFuturesSource(DataSource):
    source_id = "yfinance_futures"

    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> PriceFrame:
        try:
            raw = yf.download(
                symbol,
                start=str(start),
                end=str(end),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            print(f"[WARN] {symbol}: yfinance fetch failed ({e})")
            return PriceFrame(symbol=symbol, data=pd.DataFrame(), source=self.source_id)

        if raw.empty:
            print(f"[WARN] {symbol}: empty response from yfinance")
            return PriceFrame(symbol=symbol, data=pd.DataFrame(), source=self.source_id)

        # yfinance ≥0.2 returns MultiIndex columns when downloading a single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        raw = self._validate_frame(raw)
        available = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
        return PriceFrame(symbol=symbol, data=raw[available], source=self.source_id)
