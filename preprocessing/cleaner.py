"""
preprocessing/cleaner.py
========================
Data cleaning transformations: outlier handling, gap filling,
type coercion, and basic quality checks.

Rule: No feature engineering here — only data quality.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from contracts.interfaces import PreprocessorBase

logger = logging.getLogger(__name__)


class OutlierClipper(PreprocessorBase):
    """
    Cap extreme values using z-score or IQR method.

    Designed for price/return series where fat tails are real
    but data errors (bad ticks) must be removed.
    """

    def __init__(
        self,
        method: str = "iqr",        # "iqr" | "zscore"
        threshold: float = 4.0,     # IQR multiplier or z-score threshold
        columns: Optional[List[str]] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.columns = columns
        self._bounds: dict = {}     # fitted per-column bounds

    def fit(self, df: pd.DataFrame) -> "OutlierClipper":
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            s = df[col].dropna()
            if self.method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            else:  # zscore
                mu, sigma = s.mean(), s.std()
                lo, hi = mu - self.threshold * sigma, mu + self.threshold * sigma
            self._bounds[col] = (lo, hi)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in self._bounds.items():
            if col in df.columns:
                n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
                if n_outliers > 0:
                    logger.debug("[OutlierClipper] %s: clipping %d outliers", col, n_outliers)
                df[col] = df[col].clip(lo, hi)
        return df


class GapFiller(PreprocessorBase):
    """
    Handle missing values in time series.

    Strategy per column type:
      prices/rates  → forward fill (last known value)
      returns       → fill with 0 (no change assumption)
      indicators    → interpolate linearly
    """

    def __init__(
        self,
        price_cols: Optional[List[str]] = None,
        return_cols: Optional[List[str]] = None,
        interpolate_cols: Optional[List[str]] = None,
        max_gap: int = 5,
    ):
        self.price_cols = price_cols or []
        self.return_cols = return_cols or []
        self.interpolate_cols = interpolate_cols or []
        self.max_gap = max_gap

    def fit(self, df: pd.DataFrame) -> "GapFiller":
        return self   # stateless

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.price_cols:
            if col in df.columns:
                df[col] = df[col].ffill(limit=self.max_gap)
        for col in self.return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        for col in self.interpolate_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="time", limit=self.max_gap)
        return df


class StationarityTransformer(PreprocessorBase):
    """
    Apply differencing or log-differencing to achieve stationarity.

    Augmented Dickey-Fuller test can be run optionally to validate.

    Methods:
      log_return    : log(P_t / P_{t-1})  — for prices/rates
      pct_change    : (P_t - P_{t-1}) / P_{t-1}
      first_diff    : P_t - P_{t-1}       — for indicators
      none          : pass through
    """

    METHODS = {"log_return", "pct_change", "first_diff", "none"}

    def __init__(self, column_methods: dict[str, str]):
        """
        Parameters
        ----------
        column_methods : dict mapping column name → transform method
            e.g. {"close": "log_return", "oni": "none", "cot_net": "first_diff"}
        """
        for m in column_methods.values():
            if m not in self.METHODS:
                raise ValueError(f"Unknown method '{m}'. Choose from {self.METHODS}")
        self.column_methods = column_methods

    def fit(self, df: pd.DataFrame) -> "StationarityTransformer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, method in self.column_methods.items():
            if col not in df.columns:
                continue
            s = df[col]
            if method == "log_return":
                df[col] = np.log(s / s.shift(1))
            elif method == "pct_change":
                df[col] = s.pct_change()
            elif method == "first_diff":
                df[col] = s.diff()
            # "none" → pass through
        return df

    def run_adf_tests(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Run Augmented Dickey-Fuller test on each configured column.
        Returns dict of {col: {"statistic": ..., "pvalue": ..., "is_stationary": bool}}
        """
        from statsmodels.tsa.stattools import adfuller

        results = {}
        for col in self.column_methods:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 20:
                continue
            try:
                stat, pval, *_ = adfuller(s, autolag="AIC")
                results[col] = {
                    "statistic": round(stat, 4),
                    "pvalue": round(pval, 4),
                    "is_stationary": pval < 0.05,
                }
            except Exception as exc:
                logger.warning("ADF test failed for %s: %s", col, exc)
        return results


class TimeAligner(PreprocessorBase):
    """
    Align multiple DataFrames to a common DatetimeIndex.

    Handles:
      - different frequencies (daily vs weekly vs monthly)
      - different calendar coverage
      - reindexing with forward fill
    """

    def __init__(
        self,
        target_freq: str = "D",
        fill_method: str = "ffill",
        max_fill: int = 5,
    ):
        self.target_freq = target_freq
        self.fill_method = fill_method
        self.max_fill = max_fill
        self._common_index: Optional[pd.DatetimeIndex] = None

    def fit(self, df: pd.DataFrame) -> "TimeAligner":
        # Build business-day index spanning the data
        self._common_index = pd.bdate_range(df.index.min(), df.index.max(), freq=self.target_freq)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._common_index is None:
            raise RuntimeError("Call fit() first")
        df = df.reindex(self._common_index)
        if self.fill_method == "ffill":
            df = df.ffill(limit=self.max_fill)
        elif self.fill_method == "bfill":
            df = df.bfill(limit=self.max_fill)
        return df

    @staticmethod
    def align_many(
        frames: dict[str, pd.DataFrame],
        target_freq: str = "D",
        max_fill: int = 5,
    ) -> pd.DataFrame:
        """
        Align and join multiple DataFrames on a common daily index.
        Columns are prefixed with their frame name to avoid collisions.
        """
        if not frames:
            return pd.DataFrame()

        aligned = []
        for name, df in frames.items():
            resampled = df.resample(target_freq).last()
            resampled.columns = [f"{name}_{c}" if c != name else c for c in resampled.columns]
            aligned.append(resampled)

        combined = pd.concat(aligned, axis=1)
        combined = combined.ffill(limit=max_fill)
        return combined
