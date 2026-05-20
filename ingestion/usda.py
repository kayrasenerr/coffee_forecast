"""
ingestion/usda.py
=================
USDA Production, Supply and Distribution (PSD) ingestion for coffee.

Source: USDA Foreign Agricultural Service (FAS) PSD API
URL: https://apps.fas.usda.gov/psdonline/app/index.html#/app/downloads

Coffee green commodity code: 0711100

Key series extracted:
  production       : country-level production (1000 60-kg bags)
  exports          : export volume
  imports          : import volume
  ending_stocks    : carry-out stocks (inventory proxy)
  domestic_consumption: disappearance

API: https://apps.fas.usda.gov/psdonline/api/v3/  (free, no key)

Since the API can be unstable, this also supports CSV fallback from
manual PSD downloads.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ingestion.base import CachingDataSource, normalise_index

logger = logging.getLogger(__name__)

_USDA_PSD_API = "https://apps.fas.usda.gov/psdonline/api/v3"

# Key coffee-producing / consuming countries (ISO FAS codes)
_COFFEE_COUNTRIES = {
    "BR": "Brazil",
    "VN": "Vietnam",
    "CO": "Colombia",
    "ID": "Indonesia",
    "ET": "Ethiopia",
    "UG": "Uganda",
    "HN": "Honduras",
    "IN": "India",
    "EU": "European Union",
    "US": "United States",
}

_ATTRIBUTE_MAP = {
    1320: "production",
    1340: "exports",
    1510: "imports",
    1540: "domestic_consumption",
    1543: "ending_stocks",
}


class USDAPSDSource(CachingDataSource):
    """
    Fetch USDA PSD coffee supply/demand data.

    Returns monthly-frequency DataFrame indexed by marketing-year start date.
    Columns: country_production, country_exports, country_ending_stocks, etc.
    """

    def __init__(
        self,
        commodity_code: str = "0711100",
        countries: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.commodity_code = commodity_code
        self.countries = countries or list(_COFFEE_COUNTRIES.keys())
        self.source_id = "usda_psd_coffee"

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            import requests
        except ImportError as e:
            raise ImportError("pip install requests") from e

        url = f"{_USDA_PSD_API}/data"
        all_rows = []

        for country_code in self.countries:
            for attr_id, attr_name in _ATTRIBUTE_MAP.items():
                try:
                    params = {
                        "commodityCode": self.commodity_code,
                        "countryCode": country_code,
                        "attributeId": attr_id,
                    }
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if not data:
                        continue
                    for row in data:
                        all_rows.append({
                            "marketing_year": row.get("marketYear"),
                            "country": _COFFEE_COUNTRIES.get(country_code, country_code),
                            "country_code": country_code,
                            "attribute": attr_name,
                            "value": row.get("value"),
                        })
                except Exception as exc:
                    logger.debug("USDA PSD %s/%s failed: %s", country_code, attr_name, exc)

        if not all_rows:
            logger.warning("[USDA] No data retrieved")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        # Pivot to wide format: index=date, columns=country_attribute
        df["date"] = pd.to_datetime(df["marketing_year"].astype(str) + "-10-01")  # Oct MYstart
        df["col_name"] = df["country_code"].str.lower() + "_" + df["attribute"]
        pivot = df.pivot_table(index="date", columns="col_name", values="value", aggfunc="first")
        pivot.index = pd.DatetimeIndex(pivot.index)
        return pivot

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def load_from_csv(csv_path: str) -> pd.DataFrame:
        """
        Load from manually-downloaded USDA PSD CSV.
        CSV format: standard FAS PSD download format.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"USDA CSV not found: {csv_path}")
        df = pd.read_csv(path)
        logger.info("[USDA] Loaded %d rows from %s", len(df), csv_path)
        return df
