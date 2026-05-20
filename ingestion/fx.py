"""
ingestion/fx.py
FX rates relevant to coffee trade.

USD/BRL is the single most important FX driver for Arabica:
Brazil produces ~40% of global supply; real depreciation boosts exports
and suppresses local prices → cascades to global price.

Uses same YFinanceFuturesSource under the hood; split into separate
module so it can be replaced with a proper FX data vendor independently.
"""
from datetime import date
from ingestion.futures import YFinanceFuturesSource
from schemas.types import PriceFrame


class FXSource:
    """Thin wrapper; keeps FX concerns isolated from futures concerns."""

    def __init__(self):
        self._src = YFinanceFuturesSource()

    def fetch(self, ticker: str, start: date, end: date) -> PriceFrame:
        pf = self._src.fetch(ticker, start, end)
        pf.source = "yfinance_fx"
        # BRL=X convention in yfinance: price = USD per 1 BRL
        # We want BRL per USD (stronger USD → higher number → bearish coffee)
        if "BRL" in ticker and not pf.data.empty:
            for col in ["open", "high", "low", "close"]:
                pf.data[col] = 1.0 / pf.data[col]
        pf.currency = "BRL/USD"
        return pf
