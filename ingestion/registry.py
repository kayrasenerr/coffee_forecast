"""
ingestion/registry.py
Single entry point for all data fetching.

DATA_MODE controls the source:
  'live'      — yfinance (requires network access to Yahoo Finance)
  'csv'       — local CSV files in cfg.csv_dir
  'synthetic' — generated data (for testing / CI)

Set via environment variable:  DATA_MODE=live|csv|synthetic
Or override in DataConfig.
"""
import os
from datetime import date, timedelta
from typing import Optional

from config.settings import DataConfig
from schemas.types import PriceFrame


class DataRegistry:
    def __init__(self, cfg: DataConfig):
        self.cfg  = cfg
        self.mode = os.environ.get("DATA_MODE", cfg.data_mode)
        self._src = self._build_source()

    def _build_source(self):
        if self.mode == "live":
            from ingestion.futures import YFinanceFuturesSource
            from ingestion.fx import FXSource
            return {"futures": YFinanceFuturesSource(), "fx": FXSource()}
        elif self.mode == "csv":
            from ingestion.csv_source import CSVDataSource
            src = CSVDataSource(self.cfg.csv_dir)
            return {"all": src}
        else:  # synthetic
            from ingestion.synthetic import SyntheticCoffeeSource
            return {"synth": SyntheticCoffeeSource(seed=self.cfg.synthetic_seed)}

    def fetch_all(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> dict[str, PriceFrame]:
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=365 * self.cfg.history_years)

        print(f"[DATA] Mode={self.mode}  {start} → {end}")

        frames: dict[str, PriceFrame] = {}

        if self.mode == "live":
            from ingestion.fx import FXSource
            fs, fx = self._src["futures"], FXSource()
            frames["arabica"] = fs.fetch(self.cfg.arabica_ticker,  start, end)
            frames["robusta"] = fs.fetch(self.cfg.robusta_ticker,  start, end)
            frames["usd_brl"] = fx.fetch(self.cfg.usd_brl_ticker,  start, end)

        elif self.mode == "csv":
            src = self._src["all"]
            frames["arabica"] = src.fetch("arabica", start, end)
            frames["robusta"] = src.fetch("robusta", start, end)
            frames["usd_brl"] = src.fetch("usd_brl", start, end)

        else:  # synthetic
            synth = self._src["synth"]
            frames["arabica"] = synth.fetch_arabica(start, end)
            frames["robusta"] = PriceFrame(symbol="RB=F", data=__import__("pandas").DataFrame(), source="synthetic")
            frames["usd_brl"] = synth.fetch_usd_brl(start, end)

        self._log_summary(frames)
        return frames

    def _log_summary(self, frames: dict[str, PriceFrame]) -> None:
        for name, pf in frames.items():
            n = len(pf.data)
            if n > 0:
                span = f"{pf.data.index[0].date()} → {pf.data.index[-1].date()}"
                print(f"  {name:12s}: {n:4d} rows  {span}")
            else:
                print(f"  {name:12s}: EMPTY (skipped)")
