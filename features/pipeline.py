"""
features/pipeline.py
====================
Feature pipeline: assembles all feature transformers and produces
a unified FeatureFrame ready for modelling.

This is the primary entry point for the feature layer.
It reads from ingestion sources (or cached raw data) and writes
to the feature store.

Usage:
    from features.pipeline import FeaturePipeline
    from contracts.schemas import CoffeeVariety, DataFrequency
    from config.settings import settings
    from features.store import ParquetFeatureStore

    store = ParquetFeatureStore(settings.features_dir)
    pipeline = FeaturePipeline(store=store)
    frame = pipeline.run(
        variety=CoffeeVariety.ARABICA,
        start="2015-01-01",
        end="2024-12-31",
        frequency=DataFrequency.DAILY,
    )
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

import pandas as pd

from contracts.interfaces import FeatureStoreBase
from contracts.schemas import CoffeeVariety, DataFrequency, FeatureFrame
from features.climate_features import ClimateFeatureTransformer
from features.macro_features import MacroFeatureTransformer
from features.positioning_features import COTFeatureTransformer
from features.price_features import PriceFeatureTransformer
from config.settings import settings

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Assembles features from multiple transformers into a single FeatureFrame.

    Design:
      - Each transformer is independent and gets only its required inputs.
      - Results are joined on a common DatetimeIndex.
      - Target variable (log return) is appended last.
      - Saves to feature store if `store` is provided.
    """

    def __init__(
        self,
        store: Optional[FeatureStoreBase] = None,
        price_transformer: Optional[PriceFeatureTransformer] = None,
        climate_transformer: Optional[ClimateFeatureTransformer] = None,
        cot_transformer: Optional[COTFeatureTransformer] = None,
        macro_transformer: Optional[MacroFeatureTransformer] = None,
    ):
        self.store = store
        self.price_tfm = price_transformer or PriceFeatureTransformer()
        self.climate_tfm = climate_transformer or ClimateFeatureTransformer()
        self.cot_tfm = cot_transformer or COTFeatureTransformer()
        self.macro_tfm = macro_transformer or MacroFeatureTransformer()

    def run(
        self,
        raw_inputs: Dict[str, pd.DataFrame],
        variety: CoffeeVariety,
        frequency: DataFrequency = DataFrequency.DAILY,
        store_name: Optional[str] = None,
    ) -> FeatureFrame:
        """
        Build the full feature matrix.

        Parameters
        ----------
        raw_inputs  : dict of source-id → raw DataFrame (from ingestion)
        variety     : which coffee variety this frame represents
        frequency   : target temporal frequency
        store_name  : if given, saves result to feature store

        Returns
        -------
        FeatureFrame  with all features joined and aligned
        """
        logger.info(
            "[FeaturePipeline] Building features for %s at freq=%s",
            variety.value, frequency.value
        )

        parts: list[pd.DataFrame] = []

        # ----------------------------------------------------------------
        # Price features
        # ----------------------------------------------------------------
        price_key = f"{variety.value}_futures"
        if price_key in raw_inputs:
            price_inputs = {"prices": raw_inputs[price_key]}
            # Add opposing variety for spread
            other_key = "robusta_futures" if variety == CoffeeVariety.ARABICA else "arabica_futures"
            if other_key in raw_inputs:
                price_inputs["prices_robusta"] = raw_inputs[other_key]
            pf = self.price_tfm.compute(price_inputs, variety, frequency)
            parts.append(pf)
            logger.debug("[FeaturePipeline] Price features: %d cols", len(pf.columns))

        # ----------------------------------------------------------------
        # Climate features
        # ----------------------------------------------------------------
        climate_inputs = {}
        if "enso" in raw_inputs:
            climate_inputs["enso"] = raw_inputs["enso"]
        for k, v in raw_inputs.items():
            if k.startswith("climate_"):
                climate_inputs[k] = v
        if climate_inputs:
            cf = self.climate_tfm.compute(climate_inputs, variety, frequency)
            parts.append(cf)
            logger.debug("[FeaturePipeline] Climate features: %d cols", len(cf.columns))

        # ----------------------------------------------------------------
        # Positioning features (COT)
        # ----------------------------------------------------------------
        cot_key = f"cot_{variety.value}"
        if cot_key in raw_inputs:
            cot_inputs = {"cot": raw_inputs[cot_key]}
            posf = self.cot_tfm.compute(cot_inputs, variety, frequency)
            parts.append(posf)
            logger.debug("[FeaturePipeline] Positioning features: %d cols", len(posf.columns))

        # ----------------------------------------------------------------
        # Macro / FX features
        # ----------------------------------------------------------------
        macro_inputs = {k: v for k, v in raw_inputs.items() if k.startswith("fx_")}
        if macro_inputs:
            mf = self.macro_tfm.compute(macro_inputs, variety, frequency)
            parts.append(mf)
            logger.debug("[FeaturePipeline] Macro features: %d cols", len(mf.columns))

        # ----------------------------------------------------------------
        # Join all parts on common index
        # ----------------------------------------------------------------
        if not parts:
            raise ValueError("No features computed — check raw_inputs keys")

        combined = pd.concat(parts, axis=1)
        combined = combined.sort_index()

        # Anchor to the primary price business-day index (avoid calendar-day
        # blowup from monthly/weekly resamples inside individual transformers)
        if price_key in raw_inputs:
            bday_idx = raw_inputs[price_key].index
            if isinstance(bday_idx, pd.DatetimeIndex):
                combined = combined.reindex(bday_idx).ffill(limit=10)
        else:
            combined = combined.ffill(limit=5)

        # ----------------------------------------------------------------
        # Append target variable: next-N-day log return
        # ----------------------------------------------------------------
        if price_key in raw_inputs:
            close = raw_inputs[price_key]["close"]
            close = close.reindex(combined.index).ffill()
            import numpy as np
            for horizon in [1, 5, 10, 21]:
                combined[f"target_log_return_{horizon}d"] = np.log(
                    close.shift(-horizon) / close
                )

        # Drop rows where all features are NaN
        combined = combined.dropna(how="all")

        frame = FeatureFrame(
            variety=variety,
            frequency=frequency,
            feature_names=list(combined.columns),
            df=combined,
            description=f"{variety.value} feature matrix at {frequency.value} frequency",
        )

        if self.store and store_name:
            self.store.save(frame, store_name)

        logger.info(
            "[FeaturePipeline] Done. Shape: %s, features: %d",
            combined.shape, len([c for c in combined.columns if not c.startswith("target")])
        )
        return frame
