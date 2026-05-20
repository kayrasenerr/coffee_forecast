"""
ingestion/cot.py
================
CFTC Commitments of Traders (COT) report ingestion.

Source: CFTC publishes free weekly COT data in CSV format.
URL: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm

Coffee C (Arabica) CFTC commodity code: 083731
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from contracts.schemas import CoffeeVariety
from ingestion.base import CachingDataSource, normalise_index

logger = logging.getLogger(__name__)

# CFTC annual disaggregated COT files (legacy format, most compatible)
_CFTC_ANNUAL_URL = (
    "https://www.cftc.gov/files/dea/history/deahistfo{year}.zip"
)
_CFTC_CURRENT_URL = (
    "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
)

# Column mapping from CFTC raw format to our schema
_COT_COLUMNS = {
    "Market_and_Exchange_Names": "market_name",
    "As_of_Date_In_Form_YYMMDD": "date_raw",
    "Open_Interest_All": "open_interest",
    "NonComm_Positions_Long_All": "noncommercial_long",
    "NonComm_Positions_Short_All": "noncommercial_short",
    "Comm_Positions_Long_All": "commercial_long",
    "Comm_Positions_Short_All": "commercial_short",
    "NonRept_Positions_Long_All": "nonreportable_long",
    "NonRept_Positions_Short_All": "nonreportable_short",
}


class CFTCCOTSource(CachingDataSource):
    """
    Fetch CFTC COT legacy futures-only data for a given commodity.

    CFTC publishes annual zip files containing all commodities.
    We filter to coffee after downloading.
    """

    def __init__(
        self,
        commodity_code: str = "083731",
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.commodity_code = commodity_code
        self.variety = variety
        self.source_id = f"cftc_cot_{variety.value}"

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch one year of COT data from CFTC."""
        import zipfile

        url = _CFTC_ANNUAL_URL.format(year=year)
        logger.info("[%s] Downloading COT %d from CFTC …", self.source_id, year)

        try:
            import requests
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("[%s] Download failed for %d: %s", self.source_id, year, exc)
            return pd.DataFrame()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            csv_name = next((n for n in names if n.endswith(".csv")), None)
            if not csv_name:
                logger.warning("[%s] No CSV in zip for %d", self.source_id, year)
                return pd.DataFrame()
            with zf.open(csv_name) as f:
                raw = pd.read_csv(f, low_memory=False)

        # Filter to our commodity
        mask = raw.get("CFTC_Commodity_Code", pd.Series(dtype=str)).astype(str).str.strip() == str(self.commodity_code)
        filtered = raw[mask].copy()
        if filtered.empty:
            logger.warning("[%s] No rows for commodity %s in %d", self.source_id, self.commodity_code, year)

        return filtered

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        years = list(range(start.year, end.year + 1))
        frames = [self._fetch_year(y) for y in years]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)

        # Normalise columns
        rename = {k: v for k, v in _COT_COLUMNS.items() if k in raw.columns}
        df = raw.rename(columns=rename)

        # Parse date
        if "date_raw" in df.columns:
            df["date"] = pd.to_datetime(df["date_raw"], format="%y%m%d", errors="coerce")
        elif "Report_Date_as_MM_DD_YYYY" in raw.columns:
            df["date"] = pd.to_datetime(raw["Report_Date_as_MM_DD_YYYY"], errors="coerce")

        df = df.dropna(subset=["date"])
        df = df.set_index("date")

        keep = [v for v in _COT_COLUMNS.values() if v in df.columns and v != "date_raw"]
        df = df[keep].copy()
        df["variety"] = self.variety.value

        # Apply date filter
        df = df.loc[str(start):str(end)]
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        numeric_cols = [
            "open_interest", "noncommercial_long", "noncommercial_short",
            "commercial_long", "commercial_short",
            "nonreportable_long", "nonreportable_short",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
