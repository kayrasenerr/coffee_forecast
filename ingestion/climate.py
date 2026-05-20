"""
ingestion/climate.py
====================
Climate data ingestion for coffee-producing regions.

Sources:
  - NOAA CPC: ONI (Oceanic Niño Index) — key ENSO indicator
  - NOAA CPC: Drought Monitor (planned)
  - NASA POWER: rainfall/temperature for specific regions (planned)

ENSO is the single most important large-scale climate driver for
global coffee production (Brazil, Vietnam, Indonesia, East Africa).

ONI > +0.5 → El Niño → drought risk in Brazil/Vietnam,
             excess rain in East Africa.
ONI < -0.5 → La Niña → drought risk in East Africa/India,
             generally favorable for Brazil.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from ingestion.base import CachingDataSource, normalise_index

logger = logging.getLogger(__name__)

# NOAA ONI table (3-month running mean of ERSST.v5 SST anomalies in Niño 3.4)
_NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


class NOAAENSOSource(CachingDataSource):
    """
    Download NOAA Oceanic Niño Index (ONI) monthly data.

    Returns monthly DataFrame with columns:
      oni          : 3-month running mean SST anomaly (°C)
      enso_phase   : "el_nino" | "la_nina" | "neutral"
      season       : 3-letter season code (DJF, JFM, …)
    """

    def __init__(
        self,
        index_type: str = "ONI",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.index_type = index_type
        self.source_id = "noaa_enso_oni"

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            import requests
            resp = requests.get(_NOAA_ONI_URL, timeout=30)
            resp.raise_for_status()
            text = resp.text
        except Exception as exc:
            logger.warning("[%s] NOAA download failed: %s", self.source_id, exc)
            return pd.DataFrame()

        df = self._parse_oni(text)
        df = df.loc[str(start):str(end)]
        return df

    def _parse_oni(self, text: str) -> pd.DataFrame:
        """Parse NOAA ONI fixed-width text format."""
        rows = []
        season_to_month = {
            "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
            "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
            "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
        }
        for line in text.strip().splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0] == "SEAS":
                continue
            try:
                season, year, total, anom = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
                month = season_to_month.get(season, 1)
                dt = pd.Timestamp(year=year, month=month, day=1)
                rows.append({"date": dt, "oni": anom, "season": season})
            except (ValueError, KeyError):
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("date")
        df["enso_phase"] = df["oni"].apply(self._classify_enso)
        return df

    @staticmethod
    def _classify_enso(oni: float) -> str:
        if oni >= 0.5:
            return "el_nino"
        if oni <= -0.5:
            return "la_nina"
        return "neutral"

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        df["oni"] = pd.to_numeric(df["oni"], errors="coerce")
        return df.dropna(subset=["oni"])


class NASAPOWERSource(CachingDataSource):
    """
    NASA POWER API — daily climate parameters for a lat/lon point.

    https://power.larc.nasa.gov/api/temporal/daily/point

    Key parameters for coffee:
      PRECTOTCORR  : precipitation (mm/day)
      T2M          : temperature at 2m (°C)
      T2M_MIN      : min temperature
      T2M_MAX      : max temperature
    """

    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    # Key producing-region coordinates
    REGIONS = {
        "brazil_sul_minas":    (-21.5, -45.4),
        "brazil_cerrado":      (-18.0, -47.0),
        "ethiopia_sidama":     (6.9,   38.4),
        "ethiopia_yirgacheffe":(6.2,   38.2),
        "vietnam_dak_lak":     (12.7,  108.0),
        "uganda_bugisu":       (1.1,   34.3),
        "colombia_huila":      (2.0,   -76.0),
    }

    def __init__(
        self,
        region: str,
        parameters: Optional[list] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if region not in self.REGIONS:
            raise ValueError(f"Unknown region: {region}. Choose from {list(self.REGIONS)}")
        self.region = region
        self.lat, self.lon = self.REGIONS[region]
        self.parameters = parameters or ["PRECTOTCORR", "T2M", "T2M_MIN"]
        self.source_id = f"nasa_power_{region}"

    def _fetch_remote(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        try:
            import requests
        except ImportError as e:
            raise ImportError("pip install requests") from e

        params = {
            "parameters": ",".join(self.parameters),
            "community": "AG",
            "longitude": self.lon,
            "latitude": self.lat,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }

        resp = requests.get(self.NASA_POWER_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = {}
        for param, values in data.get("properties", {}).get("parameter", {}).items():
            records[param.lower()] = values

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "date"
        df["region"] = self.region

        # NASA uses -999 as missing
        df = df.replace(-999, float("nan"))
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = normalise_index(df)
        return df
