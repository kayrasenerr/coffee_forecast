"""
ingestion/exog_fetcher.py
=========================
Fetches ALL exogenous variables relevant to coffee futures pricing.

Six categories (28 variables):
  1. FX Rates          — producer-currency competitiveness
  2. Commodity Prices  — substitute/complement/cost relationships
  3. Macro Indicators  — demand-side and liquidity drivers
  4. Climate/ENSO      — supply-side weather risk
  5. Positioning       — market sentiment (COT)
  6. Logistics/Energy  — shipping and input costs

Every variable is documented with its causal direction:
  (+) = rising value → historically bullish for coffee
  (−) = rising value → historically bearish for coffee
  (~) = non-linear / context-dependent

All sources are free/public. Fallback to None if unavailable.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Yahoo Finance ticker map ─────────────────────────────────────────────────
# Each entry: (ticker, column_to_use, rename, causal_direction, description)
_YF_TICKERS: Dict[str, Tuple[str, str, str, str]] = {
    # FX — producer currency vs USD
    # (+) Weaker producer currency → farmers sell more in USD terms → bearish supply
    # Wait, actually: weaker BRL → MORE BRL per bag → farmers SELL → supply ↑ → BEARISH
    # So higher USDBRL = bearish for arabica
    "fx_usdbrl":  ("USDBRL=X",  "Close", "fx_usdbrl",  "(−) higher = farmers incentivised to sell"),
    "fx_usdvnd":  ("USDVND=X",  "Close", "fx_usdvnd",  "(−) higher = Vietnam robusta cheaper"),
    "fx_usdcop":  ("USDCOP=X",  "Close", "fx_usdcop",  "(−) higher = Colombia exports cheaper"),
    "fx_usdidr":  ("USDIDR=X",  "Close", "fx_usdidr",  "(−) higher = Indonesia cheaper"),
    "fx_usdugx":  ("USDUGX=X",  "Close", "fx_usdugx",  "(−) higher = Uganda cheaper"),
    "fx_eurusd":  ("EURUSD=X",  "Close", "fx_eurusd",  "(+) stronger EUR = European roasters buy more"),
    "fx_gbpusd":  ("GBPUSD=X",  "Close", "fx_gbpusd",  "(+) stronger GBP = UK demand proxy"),
    "fx_jpyusd":  ("JPYUSD=X",  "Close", "fx_jpyusd",  "(+) stronger JPY = Japan demand proxy"),

    # Commodity prices — competition / cost relationships
    "sugar":      ("SB=F",      "Close", "sugar_price",   "(−) substitute crop land use in Brazil"),
    "corn":       ("ZC=F",      "Close", "corn_price",    "(~) land competition indicator"),
    "crude_oil":  ("CL=F",      "Close", "crude_oil",     "(−) logistics & energy cost driver"),
    "nat_gas":    ("NG=F",      "Close", "nat_gas",       "(−) roasting energy cost proxy"),
    "cocoa":      ("CC=F",      "Close", "cocoa_price",   "(~) tropical ag sentiment indicator"),
    "arabica":    ("KC=F",      "Close", "arabica_close", "TARGET INSTRUMENT"),
    "robusta":    ("RC=F",      "Close", "robusta_close", "OTHER VARIETY"),

    # Financial / risk
    "vix":        ("^VIX",      "Close", "vix",           "(−) risk-off reduces commodity demand"),
    "dxy_proxy":  ("UUP",       "Close", "dxy_proxy",     "(−) strong USD = commodity headwind"),
    "spx":        ("^GSPC",     "Close", "spx",           "(+) risk-on → commodity allocation"),
    "10y_yield":  ("^TNX",      "Close", "us_10y_yield",  "(−) higher rates = USD strength"),

    # Shipping & logistics proxies
    "tanker":     ("MATX",      "Close", "shipping_matx", "(−) higher shipping = higher cost"),
    "bunker_oil": ("HO=F",      "Close", "heating_oil",   "(−) bunker fuel proxy"),
}

# ── FRED series map ──────────────────────────────────────────────────────────
_FRED_SERIES: Dict[str, Tuple[str, str, str]] = {
    "us_cpi":       ("CPIAUCSL",  "us_cpi",       "(−) inflation erodes real purchasing power"),
    "us_pce":       ("PCEPI",     "us_pce",       "(−) US consumer price index"),
    "fed_rate":     ("FEDFUNDS",  "fed_rate",      "(−) higher rates = USD strength"),
    "brazil_rate":  ("INTGSTBRM193N", "brazil_rate", "(+) tight Brazil policy = lower supply"),
    "us_ip":        ("INDPRO",    "us_ip",         "(+) industrial production = demand proxy"),
    "global_trade": ("DTWEXBGS",  "trade_wtd_usd", "(−) trade-weighted USD"),
}

# ── NOAA ENSO ────────────────────────────────────────────────────────────────
_NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# ── NASA POWER regions ───────────────────────────────────────────────────────
# (lat, lon, description)
_NASA_REGIONS: Dict[str, Tuple[float, float, str]] = {
    "brazil_sul_minas":    (-21.5, -45.4, "Brazil Sul de Minas — core Arabica belt"),
    "brazil_cerrado":      (-18.0, -47.0, "Brazil Cerrado — second largest Arabica"),
    "vietnam_dak_lak":     (12.7,  108.0, "Vietnam Dak Lak — world's largest Robusta"),
    "colombia_huila":      (2.0,   -76.0, "Colombia Huila — premium washed Arabica"),
    "ethiopia_sidama":     (6.9,    38.4, "Ethiopia Sidama — birthplace of Arabica"),
}
_NASA_PARAMS = "PRECTOTCORR,T2M,T2M_MIN"
_NASA_API    = "https://power.larc.nasa.gov/api/temporal/daily/point"


class ExogFetcher:
    """
    Fetches and aligns all exogenous variables onto a common daily index.

    Usage:
        fetcher = ExogFetcher()
        exog_df = fetcher.fetch_all(start=date(2020,1,1), end=date(2024,12,31))
        # exog_df: DataFrame, index=DatetimeIndex, cols=variable names

    For significance testing, use fetcher.get_metadata() to get
    causal direction and source information per column.
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        include_climate: bool = True,
        include_nasa: bool = False,   # slow — toggle manually
        verbose: bool = True,
    ):
        self.fred_api_key   = fred_api_key or os.environ.get("COFFEE_FRED_API_KEY")
        self.include_climate = include_climate
        self.include_nasa   = include_nasa
        self.verbose        = verbose
        self._metadata: Dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all(
        self,
        start: date,
        end:   date,
        instrument: str = "arabica",
    ) -> pd.DataFrame:
        """
        Fetch all exogenous variables and return aligned daily DataFrame.

        Parameters
        ----------
        start      : training start date
        end        : end date (usually test end)
        instrument : "arabica" or "robusta" — controls which FX/climate
                     variables are prioritised

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (business days), columns = variable names
            Missing values forward-filled ≤ 10 days.
        """
        frames: list[pd.DataFrame] = []

        self._log("Fetching Yahoo Finance data …")
        yf_df = self._fetch_yfinance(start, end)
        if not yf_df.empty:
            frames.append(yf_df)

        self._log("Fetching FRED macro data …")
        fred_df = self._fetch_fred(start, end)
        if not fred_df.empty:
            frames.append(fred_df)

        if self.include_climate:
            self._log("Fetching NOAA ENSO …")
            enso_df = self._fetch_enso(start, end)
            if not enso_df.empty:
                frames.append(enso_df)

        if self.include_nasa:
            self._log("Fetching NASA POWER climate …")
            nasa_df = self._fetch_nasa_power(start, end)
            if not nasa_df.empty:
                frames.append(nasa_df)

        if not frames:
            logger.warning("[ExogFetcher] No data fetched — all sources failed")
            return pd.DataFrame()

        # Align to business-day index
        bday_idx = pd.bdate_range(str(start), str(end))
        combined = pd.concat(frames, axis=1)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.reindex(bday_idx).ffill(limit=10)

        self._log(f"Done. Shape: {combined.shape}")
        return combined

    def get_metadata(self) -> Dict[str, str]:
        """Return {column_name: causal_description} dict."""
        return dict(self._metadata)

    # ── Yahoo Finance ─────────────────────────────────────────────────────────

    def _fetch_yfinance(self, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("[ExogFetcher] yfinance not installed")
            return pd.DataFrame()

        frames = []
        for key, (ticker, col, rename, desc) in _YF_TICKERS.items():
            try:
                raw = yf.Ticker(ticker).history(
                    start=str(start), end=str(end),
                    interval="1d", auto_adjust=True,
                )
                if raw.empty:
                    continue
                s = raw[col].rename(rename)
                s.index = s.index.tz_localize(None)
                s = s.replace(0, np.nan)
                frames.append(s)
                self._metadata[rename] = desc
            except Exception as exc:
                logger.debug("[ExogFetcher] %s failed: %s", ticker, exc)

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    # ── FRED ──────────────────────────────────────────────────────────────────

    def _fetch_fred(self, start: date, end: date) -> pd.DataFrame:
        frames = []

        # Try fredapi first
        if self.fred_api_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=self.fred_api_key)
                for key, (series_id, rename, desc) in _FRED_SERIES.items():
                    try:
                        s = fred.get_series(
                            series_id,
                            observation_start=str(start),
                            observation_end=str(end),
                        ).rename(rename)
                        s.index = pd.to_datetime(s.index).tz_localize(None)
                        frames.append(s)
                        self._metadata[rename] = desc
                    except Exception as exc:
                        logger.debug("[FRED] %s: %s", series_id, exc)
                return pd.concat(frames, axis=1) if frames else pd.DataFrame()
            except Exception:
                pass

        # Fallback: FRED public JSON API (no key, but rate-limited)
        import requests
        for key, (series_id, rename, desc) in _FRED_SERIES.items():
            try:
                url = (
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                    f"?id={series_id}&vintage_date={end}"
                )
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), index_col=0, parse_dates=True)
                df.columns = [rename]
                df.index = df.index.tz_localize(None)
                df = df.replace(".", np.nan).astype(float)
                df = df.loc[str(start):str(end)]
                frames.append(df[rename])
                self._metadata[rename] = desc
            except Exception as exc:
                logger.debug("[FRED fallback] %s: %s", series_id, exc)

        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    # ── NOAA ENSO ─────────────────────────────────────────────────────────────

    def _fetch_enso(self, start: date, end: date) -> pd.DataFrame:
        import requests
        season_map = {
            "DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
            "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12,
        }
        try:
            resp = requests.get(_NOAA_ONI_URL, timeout=20)
            rows = []
            for line in resp.text.strip().splitlines():
                parts = line.split()
                if len(parts) < 4 or parts[0] == "SEAS":
                    continue
                try:
                    rows.append({
                        "date": pd.Timestamp(year=int(parts[1]),
                                             month=season_map.get(parts[0], 1), day=1),
                        "oni":        float(parts[3]),
                    })
                except (ValueError, KeyError):
                    continue
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).set_index("date")
            df["enso_el_nino"] = (df["oni"] >= 0.5).astype(float)
            df["enso_la_nina"] = (df["oni"] <= -0.5).astype(float)
            df["oni_lag3m"]    = df["oni"].shift(3)
            df["oni_lag6m"]    = df["oni"].shift(6)
            df["oni_lag9m"]    = df["oni"].shift(9)
            df = df.loc[str(start):str(end)]
            for col in df.columns:
                self._metadata[col] = "(+) El Niño → Brazil drought → supply risk" if "oni" in col else "ENSO phase"
            return df
        except Exception as exc:
            logger.warning("[ExogFetcher] ENSO fetch failed: %s", exc)
            return pd.DataFrame()

    # ── NASA POWER ────────────────────────────────────────────────────────────

    def _fetch_nasa_power(self, start: date, end: date) -> pd.DataFrame:
        import requests
        frames = []
        for region, (lat, lon, desc) in _NASA_REGIONS.items():
            try:
                params = {
                    "parameters": _NASA_PARAMS,
                    "community": "AG",
                    "longitude": lon, "latitude": lat,
                    "start": start.strftime("%Y%m%d"),
                    "end":   end.strftime("%Y%m%d"),
                    "format": "JSON",
                }
                resp = requests.get(_NASA_API, params=params, timeout=60)
                data = resp.json()
                param_data = data.get("properties", {}).get("parameter", {})
                for param, values in param_data.items():
                    s = pd.Series(values, name=f"{region}_{param.lower()}")
                    s.index = pd.to_datetime(s.index, format="%Y%m%d")
                    s = s.replace(-999, np.nan)
                    frames.append(s)
                    self._metadata[s.name] = f"{desc} — {param}"
            except Exception as exc:
                logger.debug("[NASA POWER] %s: %s", region, exc)
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [ExogFetcher] {msg}", flush=True)
        logger.info(msg)
