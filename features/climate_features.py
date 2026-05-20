"""
features/climate_features.py
=============================
Climate-derived feature engineering.

Key causal relationships modelled:
  ENSO (ONI) → Brazil/Vietnam rainfall → production → price
  Frost events in Brazil → immediate supply shock → price spike
  Drought SPI → lagged 6-12 months → harvest volume impact

Feature philosophy:
  - Use anomalies (deviation from climatological baseline)
  - Include lagged versions (price responds with delay)
  - Flag threshold exceedances (El Niño onset, drought declared)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from contracts.interfaces import FeatureTransformerBase
from contracts.schemas import CoffeeVariety, DataFrequency

logger = logging.getLogger(__name__)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sigma.replace(0, np.nan)


class ClimateFeatureTransformer(FeatureTransformerBase):
    """
    Compute climate-based features from ENSO and regional climate data.

    Required inputs:
      "enso"   : DataFrame with 'oni' column (monthly)

    Optional inputs:
      "climate_{region}" : NASA POWER DataFrames per region
    """

    feature_group = "climate"

    def __init__(
        self,
        include_enso: bool = True,
        enso_lags: Optional[List[int]] = None,
        include_regional: bool = False,
    ):
        self.include_enso = include_enso
        self.enso_lags = enso_lags or [0, 1, 3, 6, 9, 12]  # months lag
        self.include_regional = include_regional

    def get_required_inputs(self) -> List[str]:
        return ["enso"]

    @property
    def output_columns(self) -> List[str]:
        cols = []
        if self.include_enso:
            cols += ["oni", "enso_is_el_nino", "enso_is_la_nina"]
            cols += [f"oni_lag_{lag}m" for lag in self.enso_lags if lag > 0]
            cols += ["oni_trend_3m", "oni_z_12m"]
        return cols

    def compute(
        self,
        inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> pd.DataFrame:
        features_parts = []

        if self.include_enso and "enso" in inputs:
            enso_features = self._compute_enso_features(inputs["enso"])
            features_parts.append(enso_features)

        if self.include_regional:
            for key, df in inputs.items():
                if key.startswith("climate_"):
                    region = key.replace("climate_", "")
                    regional_features = self._compute_regional_features(df, region)
                    features_parts.append(regional_features)

        if not features_parts:
            return pd.DataFrame()

        # Combine and forward-fill to daily frequency
        combined = pd.concat(features_parts, axis=1)
        combined = combined.resample(frequency.value).last().ffill(limit=31)
        return combined

    def _compute_enso_features(self, enso_df: pd.DataFrame) -> pd.DataFrame:
        """Process NOAA ONI data into climate features."""
        f = pd.DataFrame(index=enso_df.index)
        oni = enso_df["oni"]

        # Raw ONI value
        f["oni"] = oni

        # Phase flags
        f["enso_is_el_nino"] = (oni >= 0.5).astype(float)
        f["enso_is_la_nina"] = (oni <= -0.5).astype(float)
        f["enso_intensity"] = oni.abs()   # strength regardless of phase

        # Consecutive months in phase (persistence = greater impact)
        f["el_nino_months"] = self._consecutive_count(oni >= 0.5)
        f["la_nina_months"] = self._consecutive_count(oni <= -0.5)

        # Lagged ONI (production lags climate by months)
        for lag in self.enso_lags:
            if lag > 0:
                f[f"oni_lag_{lag}m"] = oni.shift(lag)

        # ONI trend (acceleration of anomaly)
        f["oni_trend_3m"] = oni.diff(3)

        # Anomaly vs long-run average
        f["oni_z_12m"] = _rolling_zscore(oni, 12)

        # Brazil-specific ENSO impact (Arabica)
        # El Niño → drought risk in Minas Gerais → supply concern
        if variety_relevant := True:  # placeholder for variety-specific logic
            f["brazil_drought_risk"] = np.where(
                oni >= 1.0, 2.0,
                np.where(oni >= 0.5, 1.0, 0.0)
            )
            # La Niña → frost risk in South Brazil (cold air intrusions)
            f["brazil_frost_risk"] = np.where(
                oni <= -1.0, 2.0,
                np.where(oni <= -0.5, 1.0, 0.0)
            )

        return f

    def _compute_regional_features(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Compute anomaly features for a producing region."""
        f = pd.DataFrame(index=df.index)
        prefix = region.replace("-", "_")

        if "prectotcorr" in df.columns:
            rain = df["prectotcorr"]
            f[f"{prefix}_rainfall_mm"] = rain
            # Anomaly: z-score vs 5-year rolling baseline
            f[f"{prefix}_rainfall_anomaly"] = _rolling_zscore(rain, 365 * 5)
            # SPI-like drought indicator (negative = dry)
            f[f"{prefix}_dry_flag"] = (_rolling_zscore(rain, 30) < -1.5).astype(float)

        if "t2m" in df.columns:
            temp = df["t2m"]
            f[f"{prefix}_temp_c"] = temp
            f[f"{prefix}_temp_anomaly"] = _rolling_zscore(temp, 365 * 5)

        if "t2m_min" in df.columns:
            t_min = df["t2m_min"]
            # Frost risk: min temp < 2°C in producing areas
            f[f"{prefix}_frost_risk"] = (t_min < 2.0).astype(float)
            f[f"{prefix}_severe_frost"] = (t_min < -2.0).astype(float)

        return f

    @staticmethod
    def _consecutive_count(condition: pd.Series) -> pd.Series:
        """Count consecutive True values (resets on False)."""
        result = pd.Series(0, index=condition.index)
        count = 0
        for i, val in enumerate(condition):
            if val:
                count += 1
            else:
                count = 0
            result.iloc[i] = count
        return result
