"""
ingestion/csv_source.py
Plug in real data via CSV files when live APIs are unavailable.

Expected CSV format (one file per symbol):
  Date,Open,High,Low,Close,Volume
  2020-01-02,129.5,131.2,128.8,130.1,12500

Usage in DataRegistry (set in config or env):
  DATA_SOURCE=csv
  CSV_DIR=/path/to/your/data/

File naming convention:
  arabica.csv   → Arabica KC=F
  robusta.csv   → Robusta RB=F
  usd_brl.csv   → USD/BRL FX rate

Minimum required columns: Date, Close
Optional: Open, High, Low, Volume (filled with Close if absent)
"""
import pandas as pd
from pathlib import Path
from datetime import date
from ingestion.base import DataSource
from schemas.types import PriceFrame


class CSVDataSource(DataSource):
    source_id = "csv"

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def fetch(self, symbol: str, start: date, end: date, **kwargs) -> PriceFrame:
        """
        symbol: logical name matching filename stem (e.g. 'arabica' → arabica.csv)
        """
        path = self.data_dir / f"{symbol}.csv"
        if not path.exists():
            print(f"[WARN] CSV not found: {path}")
            return PriceFrame(symbol=symbol, data=pd.DataFrame(), source=self.source_id)

        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df = self._validate_frame(df)

        # Fill optional columns if missing
        for col in ["open", "high", "low", "volume"]:
            if col not in df.columns:
                df[col] = df["close"]

        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df = df[mask]

        if df.empty:
            print(f"[WARN] {symbol}: CSV exists but no data in [{start}, {end}]")

        return PriceFrame(symbol=symbol, data=df[["open","high","low","close","volume"]],
                          source=self.source_id)
