"""
contracts/schemas.py
====================
Canonical data schemas for the Coffee Quant system.

ALL data crossing module boundaries must conform to these types.
This file is the single source of truth for data shapes.
Never import from other coffee_quant modules here.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CoffeeVariety(str, Enum):
    ARABICA = "arabica"
    ROBUSTA = "robusta"


class Exchange(str, Enum):
    ICE_NEW_YORK = "ICE_NY"    # Arabica (KC)
    ICE_LONDON = "ICE_LON"     # Robusta (RC)


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CONTANGO = "contango"       # Futures curve in contango
    BACKWARDATION = "backwardation"
    VOLATILE = "volatile"
    LOW_VOL = "low_vol"
    SUPPLY_STRESS = "supply_stress"
    NEUTRAL = "neutral"


class DataFrequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"


class DataSource(str, Enum):
    ICE = "ICE"
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "FRED"
    NOAA = "NOAA"
    USDA = "USDA"
    CFTC = "CFTC"
    ICO = "ICO"
    SYNTHETIC = "synthetic"     # Generated / simulated data


# ---------------------------------------------------------------------------
# Core time-series record
# ---------------------------------------------------------------------------

class OHLCVRecord(BaseModel):
    """Single OHLCV bar for a futures contract."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    source: DataSource = DataSource.SYNTHETIC
    variety: Optional[CoffeeVariety] = None
    exchange: Optional[Exchange] = None

    @field_validator("close", "open", "high", "low")
    @classmethod
    def positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class FuturesCurveRecord(BaseModel):
    """Snapshot of the futures curve (term structure) at a point in time."""
    timestamp: datetime
    variety: CoffeeVariety
    # tenor -> settlement price  e.g. {"M1": 185.5, "M2": 187.0, ...}
    tenors: Dict[str, float]
    # Derived spreads (optional, computed by feature pipeline)
    front_back_spread: Optional[float] = None   # M1 - M2
    carry_annualized: Optional[float] = None


class FXRecord(BaseModel):
    """FX spot rate record."""
    timestamp: datetime
    pair: str           # e.g. "USDBRL"
    rate: float
    source: DataSource = DataSource.FRED


class ClimateRecord(BaseModel):
    """Climate observation or anomaly for a producing region."""
    timestamp: datetime
    region: str         # e.g. "brazil_sul_minas", "ethiopia_sidama"
    variable: str       # e.g. "rainfall_mm", "temp_c", "spi_3m"
    value: float
    anomaly: Optional[float] = None     # deviation from climatological mean
    source: DataSource = DataSource.NOAA


class InventoryRecord(BaseModel):
    """Certified stock / warehouse inventory snapshot."""
    timestamp: datetime
    exchange: Exchange
    variety: CoffeeVariety
    certified_lots: float
    pending_lots: Optional[float] = None
    yoy_change_pct: Optional[float] = None


class COTRecord(BaseModel):
    """CFTC Commitments of Traders snapshot."""
    timestamp: datetime
    variety: CoffeeVariety
    commercial_long: float
    commercial_short: float
    noncommercial_long: float
    noncommercial_short: float
    nonreportable_long: float
    nonreportable_short: float

    @property
    def net_noncommercial(self) -> float:
        return self.noncommercial_long - self.noncommercial_short

    @property
    def commercial_net(self) -> float:
        return self.commercial_long - self.commercial_short


class ExportRecord(BaseModel):
    """Coffee export statistic for a producing country."""
    timestamp: datetime
    country: str        # ISO-3166 alpha-3 or common name
    variety: Optional[CoffeeVariety] = None
    volume_60kg_bags: float
    yoy_change_pct: Optional[float] = None
    source: DataSource = DataSource.ICO


# ---------------------------------------------------------------------------
# Feature store record  (output of feature engineering layer)
# ---------------------------------------------------------------------------

class FeatureVector(BaseModel):
    """
    A named vector of engineered features at a single timestamp.
    This is the canonical input to all models.
    """
    timestamp: datetime
    features: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_series(self) -> pd.Series:
        s = pd.Series(self.features, name=self.timestamp)
        return s


class FeatureFrame(BaseModel):
    """
    A DataFrame-backed collection of FeatureVectors over time.
    Wraps pd.DataFrame with validated schema metadata.
    """
    model_config = {"arbitrary_types_allowed": True}

    variety: CoffeeVariety
    frequency: DataFrequency
    feature_names: List[str]
    df: pd.DataFrame          # index=DatetimeIndex, cols=feature_names
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""

    @field_validator("df")
    @classmethod
    def validate_df_index(cls, v: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("FeatureFrame.df must have a DatetimeIndex")
        return v


# ---------------------------------------------------------------------------
# Model output schemas
# ---------------------------------------------------------------------------

class RegimeSnapshot(BaseModel):
    """Point-in-time regime assignment from HMM or rule-based classifier."""
    timestamp: datetime
    variety: CoffeeVariety
    regime: MarketRegime
    regime_id: int              # raw numeric state from HMM
    probability: float          # P(regime | observations)
    transition_probability: Optional[float] = None   # P(regime change next step)


class ForecastRecord(BaseModel):
    """Probabilistic price/return forecast."""
    generated_at: datetime
    forecast_horizon_days: int
    variety: CoffeeVariety
    # point estimates
    mean_return: float
    median_return: float
    # quantiles
    q10: float
    q25: float
    q75: float
    q90: float
    # directional probability
    prob_up: float              # P(return > 0)
    prob_large_move: float      # P(|return| > threshold)
    # model metadata
    model_name: str
    regime_at_forecast: Optional[MarketRegime] = None
    confidence: Optional[float] = None


class VolatilityForecast(BaseModel):
    """GARCH-style volatility forecast."""
    generated_at: datetime
    variety: CoffeeVariety
    horizon_days: int
    forecast_annualized_vol: float
    current_annualized_vol: float
    vol_regime: str             # "low" | "medium" | "high" | "extreme"
    model_name: str


# ---------------------------------------------------------------------------
# Backtesting output
# ---------------------------------------------------------------------------

class BacktestResult(BaseModel):
    """Aggregated result from a walk-forward backtest run."""
    model_name: str
    variety: CoffeeVariety
    start_date: date
    end_date: date
    n_folds: int
    # directional accuracy
    directional_accuracy: float
    # calibration
    mean_brier_score: Optional[float] = None
    # continuous forecast metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    # regime detection
    regime_transition_f1: Optional[float] = None
    # sharpe of signal-based returns (if applicable)
    signal_sharpe: Optional[float] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
