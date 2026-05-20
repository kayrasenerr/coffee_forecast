"""
contracts/interfaces.py
=======================
Abstract Base Classes (ABCs) for every pluggable component.

Concrete implementations MUST inherit from these.
This guarantees that:
  - modules are interchangeable,
  - the orchestrator can work with any implementation,
  - future AI-assisted development has clear contracts to implement.

Rule: Never import concrete implementations here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from contracts.schemas import (
    BacktestResult,
    CoffeeVariety,
    DataFrequency,
    FeatureFrame,
    ForecastRecord,
    RegimeSnapshot,
    VolatilityForecast,
)


# ---------------------------------------------------------------------------
# Data Ingestion Interface
# ---------------------------------------------------------------------------

class DataSourceBase(ABC):
    """
    Interface for all data ingestion adapters.

    Each concrete source (futures, FX, climate, COT, etc.) implements this.
    The source is responsible for fetching, light validation, and returning
    a standardised DataFrame. It does NOT perform feature engineering.
    """

    source_id: str          # Unique identifier, e.g. "yahoo_futures_arabica"

    @abstractmethod
    def fetch(
        self,
        start: date,
        end: date,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch raw data for the given date range.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (UTC-normalised)
            Columns: source-specific but documented in each implementation
        """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Light validation / type coercion.  Raises ValueError on fatal issues.
        Returns cleaned DataFrame.
        """

    def fetch_validated(self, start: date, end: date, **kwargs: Any) -> pd.DataFrame:
        """Convenience: fetch + validate in one call."""
        return self.validate(self.fetch(start, end, **kwargs))


# ---------------------------------------------------------------------------
# Preprocessing Interface
# ---------------------------------------------------------------------------

class PreprocessorBase(ABC):
    """Transform a raw DataFrame into a clean, aligned, analysis-ready form."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "PreprocessorBase":
        """Learn any statistics needed (e.g. mean/std for normalisation)."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation.  Stateless after fit()."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Feature Engineering Interface
# ---------------------------------------------------------------------------

class FeatureTransformerBase(ABC):
    """
    Computes a named set of features from one or more input DataFrames.

    Each concrete transformer is responsible for a cohesive feature group
    (e.g. price features, climate anomalies, COT positioning).
    """

    feature_group: str      # e.g. "price", "climate", "positioning"
    output_columns: List[str]   # declared output column names

    @abstractmethod
    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        """
        Compute features from named input DataFrames.

        Parameters
        ----------
        inputs : dict mapping string keys → DataFrames
            Keys are source identifiers ("prices", "cot", "climate", etc.)
        variety : CoffeeVariety
        frequency : DataFrequency

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns = self.output_columns (subset allowed)
        """

    def get_required_inputs(self) -> List[str]:
        """Return list of required keys in the `inputs` dict."""
        return []


# ---------------------------------------------------------------------------
# Feature Store Interface
# ---------------------------------------------------------------------------

class FeatureStoreBase(ABC):
    """Persistent storage and retrieval of computed feature frames."""

    @abstractmethod
    def save(self, frame: FeatureFrame, name: str) -> None:
        """Persist a FeatureFrame under the given name."""

    @abstractmethod
    def load(self, name: str) -> FeatureFrame:
        """Load a FeatureFrame by name.  Raises KeyError if not found."""

    @abstractmethod
    def list_available(self) -> List[str]:
        """Return names of all persisted FeatureFrames."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check whether a FeatureFrame exists without loading it."""


# ---------------------------------------------------------------------------
# Model Interface
# ---------------------------------------------------------------------------

class ModelBase(ABC):
    """
    Interface for all statistical/ML models in the system.

    Separation:
    - fit()     → trains on historical FeatureFrame
    - predict() → produces ForecastRecord(s)
    - Models must be serialisable (save/load).
    """

    model_name: str
    variety: CoffeeVariety

    @abstractmethod
    def fit(self, frame: FeatureFrame, target_col: str) -> "ModelBase":
        """
        Fit model on FeatureFrame.

        Parameters
        ----------
        frame      : FeatureFrame  (features aligned to target)
        target_col : name of the target column in frame.df
        """

    @abstractmethod
    def predict(
        self,
        frame: FeatureFrame,
        horizon_days: int = 5,
    ) -> ForecastRecord:
        """Generate probabilistic forecast from latest available features."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise model to disk."""

    @abstractmethod
    def load(self, path: str) -> "ModelBase":
        """Deserialise model from disk."""

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameter dict for experiment tracking."""
        return {}


# ---------------------------------------------------------------------------
# Regime Detector Interface
# ---------------------------------------------------------------------------

class RegimeDetectorBase(ABC):
    """
    Interface for regime detection models (e.g. HMM, rule-based classifiers).
    """

    model_name: str
    n_regimes: int

    @abstractmethod
    def fit(self, frame: FeatureFrame) -> "RegimeDetectorBase":
        """Fit regime model on FeatureFrame (unsupervised or semi-supervised)."""

    @abstractmethod
    def predict(self, frame: FeatureFrame) -> List[RegimeSnapshot]:
        """Return regime assignment for each row in frame."""

    @abstractmethod
    def predict_latest(self, frame: FeatureFrame) -> RegimeSnapshot:
        """Return regime for the most recent observation."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> "RegimeDetectorBase": ...


# ---------------------------------------------------------------------------
# Volatility Model Interface
# ---------------------------------------------------------------------------

class VolatilityModelBase(ABC):

    model_name: str

    @abstractmethod
    def fit(self, returns: pd.Series) -> "VolatilityModelBase":
        """Fit on log-return series."""

    @abstractmethod
    def forecast(
        self,
        returns: pd.Series,
        horizon_days: int = 10,
    ) -> VolatilityForecast:
        """Return volatility forecast."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> "VolatilityModelBase": ...


# ---------------------------------------------------------------------------
# Backtesting Interface
# ---------------------------------------------------------------------------

class BacktesterBase(ABC):
    """Walk-forward validation engine."""

    @abstractmethod
    def run(
        self,
        model: ModelBase,
        frame: FeatureFrame,
        target_col: str,
        n_folds: int,
        train_window: int,
        test_window: int,
    ) -> BacktestResult:
        """Execute walk-forward backtest and return aggregated metrics."""


# ---------------------------------------------------------------------------
# Experiment Tracker Interface
# ---------------------------------------------------------------------------

class ExperimentTrackerBase(ABC):
    """Lightweight interface for logging experiments."""

    @abstractmethod
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Begin a new experiment run; return run_id."""

    @abstractmethod
    def log_params(self, run_id: str, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, run_id: str, path: str) -> None: ...

    @abstractmethod
    def end_run(self, run_id: str) -> None: ...
