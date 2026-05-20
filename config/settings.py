"""
config/settings.py
==================
Global configuration for the Coffee Quant system.

Environment variables override defaults. Prefix: COFFEE_
Example: COFFEE_DATA_DIR=/mnt/data

Usage:
    from config.settings import settings
    print(settings.data_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COFFEE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    project_root: Path = Field(default=Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "data" / "models"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "data" / "experiments"

    # -----------------------------------------------------------------------
    # API keys (set via env or .env file — never commit)
    # -----------------------------------------------------------------------
    fred_api_key: Optional[str] = Field(default=None)
    quandl_api_key: Optional[str] = Field(default=None)
    noaa_api_token: Optional[str] = Field(default=None)

    # -----------------------------------------------------------------------
    # Data defaults
    # -----------------------------------------------------------------------
    default_start_date: str = "2010-01-01"
    default_end_date: str = "2024-12-31"
    default_frequency: str = "D"

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    # Lookback windows (in bars at default frequency)
    short_window: int = 10
    medium_window: int = 30
    long_window: int = 90
    extra_long_window: int = 252   # ~1 year of trading days

    # Anomaly z-score threshold for flagging
    anomaly_zscore_threshold: float = 2.0

    # -----------------------------------------------------------------------
    # HMM / Regime
    # -----------------------------------------------------------------------
    hmm_n_regimes: int = 3
    hmm_n_iter: int = 200
    hmm_covariance_type: str = "full"

    # -----------------------------------------------------------------------
    # GARCH
    # -----------------------------------------------------------------------
    garch_p: int = 1
    garch_q: int = 1
    garch_dist: str = "t"          # Student-t for fat tails

    # -----------------------------------------------------------------------
    # SARIMAX
    # -----------------------------------------------------------------------
    sarimax_order: tuple = (1, 1, 1)
    sarimax_seasonal_order: tuple = (1, 1, 1, 52)  # weekly seasonality

    # -----------------------------------------------------------------------
    # Backtesting
    # -----------------------------------------------------------------------
    backtest_n_folds: int = 5
    backtest_train_window_days: int = 756    # ~3 years
    backtest_test_window_days: int = 63      # ~3 months
    backtest_step_days: int = 63

    # -----------------------------------------------------------------------
    # Producing regions (used by climate / export modules)
    # -----------------------------------------------------------------------
    arabica_regions: List[str] = [
        "brazil_sul_minas",
        "brazil_cerrado",
        "brazil_mogiana",
        "ethiopia_sidama",
        "ethiopia_yirgacheffe",
        "colombia_huila",
    ]

    robusta_regions: List[str] = [
        "vietnam_dak_lak",
        "uganda_bugisu",
        "ivory_coast",
        "indonesia_sumatra",
    ]

    # -----------------------------------------------------------------------
    # FX pairs relevant to coffee trade
    # -----------------------------------------------------------------------
    fx_pairs: List[str] = [
        "USDBRL",   # Brazil
        "USDVND",   # Vietnam
        "USDETH",   # Ethiopia (unofficial proxy)
        "USDUGX",   # Uganda
        "USDCOP",   # Colombia
        "EURUSD",   # European demand proxy
    ]

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()


def ensure_dirs() -> None:
    """Create all required data directories if they don't exist."""
    for d in [
        settings.raw_dir,
        settings.processed_dir,
        settings.features_dir,
        settings.models_dir,
        settings.experiments_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
