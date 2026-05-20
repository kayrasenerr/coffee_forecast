"""
ingestion/ico.py
================
International Coffee Organization (ICO) export statistics ingestion.

The ICO publishes monthly coffee trade statistics by:
  - Exporting country
  - Coffee type (Arabica Milds, Brazilian Naturals, Robustas, Other Arabica)
  - Volume (60-kg bags)

Data access:
  ICO requires data files to be downloaded manually from:
  https://www.ico.org/new_historical.asp

  This module supports:
  1. Manual CSV/Excel download loading (primary)
  2. Scraped monthly press release parsing (secondary, fragile)

Key series for coffee quant:
  - Brazil exports (Arabica supply signal)
  - Vietnam exports (Robusta supply signal)
  - Colombia exports (Mild Arabica signal)
  - Ethiopia exports (Washed Arabica signal)
  - Total world exports (broad supply)
  - Shipment vs production gap (inventory build/draw signal)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from contracts.schemas import CoffeeVariety, ExportRecord
from ingestion.base import CachingDataSource, normalise_index

logger = logging.getLogger(__name__)

# Countries with strong Arabica / Robusta signal relevance
_ARABICA_EXPORTERS = ["Brazil", "Colombia", "Ethiopia", "Honduras", "Guatemala"]
_ROBUSTA_EXPORTERS = ["Vietnam", "Indonesia", "Uganda", "Ivory Coast"]

# ICO coffee type → variety mapping
_ICO_TYPE_VARIETY = {
    "Brazilian Naturals": CoffeeVariety.ARABICA,
    "Colombian Milds": CoffeeVariety.ARABICA,
    "Other Milds": CoffeeVariety.ARABICA,
    "Robustas": CoffeeVariety.ROBUSTA,
}


class ICOExportSource(CachingDataSource):
    """
    Load ICO coffee export data from manually-downloaded files.

    Supports CSV and Excel formats from the ICO historical data page.
    Cannot auto-download without authentication.

    Usage:
        source = ICOExportSource(data_path="/path/to/ico_exports.csv")
        df = source.fetch_validated(start, end)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.data_path = data_path
        self.source_id = "ico_exports"

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        if not self.data_path:
            logger.warning(
                "[ICO] No data_path configured. Download from "
                "https://www.ico.org/new_historical.asp and set ico_data_path."
            )
            return pd.DataFrame()

        path = Path(self.data_path)
        if not path.exists():
            logger.warning("[ICO] Data file not found: %s", self.data_path)
            return pd.DataFrame()

        if path.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(path, sheet_name=0)
        else:
            raw = pd.read_csv(path)

        return self._parse_ico_format(raw, start, end)

    def _parse_ico_format(
        self, raw: pd.DataFrame, start: date, end: date
    ) -> pd.DataFrame:
        """
        Parse ICO standard export format.
        ICO files typically have: Country, Type, Year, Month1..Month12 or Date, Volume
        """
        raw.columns = [str(c).strip() for c in raw.columns]
        rows = []

        # Wide format: Country, Type, [year-month columns]
        id_cols = {"Country", "country", "Exporting Country"}
        type_cols = {"Type", "type", "Coffee Type"}
        date_cols = [c for c in raw.columns if c not in id_cols and c not in type_cols]

        country_col = next((c for c in id_cols if c in raw.columns), None)
        type_col = next((c for c in type_cols if c in raw.columns), None)

        for _, row in raw.iterrows():
            country = str(row.get(country_col, "Unknown")).strip() if country_col else "Unknown"
            coffee_type = str(row.get(type_col, "")).strip() if type_col else ""
            variety = _ICO_TYPE_VARIETY.get(coffee_type)

            for col in date_cols:
                try:
                    dt = pd.to_datetime(col, errors="coerce")
                    if pd.isna(dt):
                        continue
                    val = pd.to_numeric(row[col], errors="coerce")
                    if pd.isna(val):
                        continue
                    rows.append({
                        "date": dt,
                        "country": country,
                        "coffee_type": coffee_type,
                        "variety": variety.value if variety else None,
                        "volume_60kg_bags": val,
                    })
                except Exception:
                    continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index("date")
        df = df.loc[str(start):str(end)]
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        df["volume_60kg_bags"] = pd.to_numeric(df["volume_60kg_bags"], errors="coerce")
        return df.dropna(subset=["volume_60kg_bags"])

    def get_country_exports(
        self,
        df: pd.DataFrame,
        country: str,
        variety: Optional[CoffeeVariety] = None,
    ) -> pd.Series:
        """Extract monthly export series for a single country."""
        mask = df["country"].str.lower() == country.lower()
        if variety:
            mask &= df["variety"] == variety.value
        country_df = df[mask]
        return country_df.groupby(level=0)["volume_60kg_bags"].sum()

    def world_total_exports(self, df: pd.DataFrame) -> pd.Series:
        """Monthly world total export volume."""
        return df.groupby(level=0)["volume_60kg_bags"].sum()
