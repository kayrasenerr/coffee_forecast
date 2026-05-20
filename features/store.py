"""
features/store.py
=================
Parquet-backed feature store.

The feature store is the canonical boundary between ingestion/feature-engineering
and modeling. Models always read from the store; they never call ingestion directly.

Layout on disk:
  {features_dir}/{name}.parquet         — the feature matrix
  {features_dir}/{name}.meta.json       — FeatureFrame metadata

Usage:
    from features.store import ParquetFeatureStore
    from config.settings import settings

    store = ParquetFeatureStore(settings.features_dir)
    store.save(frame, "arabica_features_D")
    frame = store.load("arabica_features_D")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from contracts.interfaces import FeatureStoreBase
from contracts.schemas import CoffeeVariety, DataFrequency, FeatureFrame

logger = logging.getLogger(__name__)


class ParquetFeatureStore(FeatureStoreBase):
    """Persist FeatureFrames as parquet files with JSON metadata sidecar."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.parquet"

    def _meta_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.meta.json"

    def save(self, frame: FeatureFrame, name: str) -> None:
        parquet_path = self._parquet_path(name)
        meta_path = self._meta_path(name)

        # Save DataFrame
        frame.df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

        # Save metadata
        meta = {
            "name": name,
            "variety": frame.variety.value,
            "frequency": frame.frequency.value,
            "feature_names": frame.feature_names,
            "created_at": frame.created_at.isoformat(),
            "description": frame.description,
            "n_rows": len(frame.df),
            "date_min": frame.df.index.min().isoformat() if not frame.df.empty else None,
            "date_max": frame.df.index.max().isoformat() if not frame.df.empty else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "[FeatureStore] Saved '%s' (%d rows, %d features)",
            name, len(frame.df), len(frame.feature_names)
        )

    def load(self, name: str) -> FeatureFrame:
        parquet_path = self._parquet_path(name)
        meta_path = self._meta_path(name)

        if not parquet_path.exists():
            raise KeyError(f"Feature store: '{name}' not found at {parquet_path}")

        df = pd.read_parquet(parquet_path, engine="pyarrow")

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        frame = FeatureFrame(
            variety=CoffeeVariety(meta.get("variety", "arabica")),
            frequency=DataFrequency(meta.get("frequency", "D")),
            feature_names=meta.get("feature_names", list(df.columns)),
            df=df,
            created_at=datetime.fromisoformat(meta["created_at"]) if "created_at" in meta else datetime.utcnow(),
            description=meta.get("description", ""),
        )

        logger.info(
            "[FeatureStore] Loaded '%s' (%d rows, %d features)",
            name, len(df), len(frame.feature_names)
        )
        return frame

    def list_available(self) -> List[str]:
        return [
            p.stem
            for p in sorted(self.base_dir.glob("*.parquet"))
        ]

    def exists(self, name: str) -> bool:
        return self._parquet_path(name).exists()

    def get_metadata(self, name: str) -> dict:
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def delete(self, name: str) -> None:
        for path in [self._parquet_path(name), self._meta_path(name)]:
            if path.exists():
                path.unlink()
        logger.info("[FeatureStore] Deleted '%s'", name)

    def summary(self) -> pd.DataFrame:
        """Return DataFrame summarising all stored feature sets."""
        rows = []
        for name in self.list_available():
            meta = self.get_metadata(name)
            rows.append({
                "name": name,
                "variety": meta.get("variety"),
                "frequency": meta.get("frequency"),
                "n_rows": meta.get("n_rows"),
                "n_features": len(meta.get("feature_names", [])),
                "date_min": meta.get("date_min"),
                "date_max": meta.get("date_max"),
                "created_at": meta.get("created_at"),
            })
        return pd.DataFrame(rows)
