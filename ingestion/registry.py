"""
ingestion/registry.py
=====================
Source registry: loads source configurations from sources.yaml
and provides factory access to DataSourceBase instances.

Usage:
    from ingestion.registry import source_registry
    source = source_registry.get("arabica_futures")
    df = source.fetch_validated(start, end)
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from contracts.interfaces import DataSourceBase
from config.settings import settings

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "sources.yaml"


class SourceRegistry:
    """
    Lazy-loading registry of data source adapters.

    Reads config/sources.yaml, instantiates adapters on first access.
    """

    def __init__(self, config_path: Path = _CONFIG_PATH):
        self._config_path = config_path
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, DataSourceBase] = {}
        self._loaded = False

    def _load_config(self) -> None:
        if self._loaded:
            return
        with open(self._config_path) as f:
            data = yaml.safe_load(f)
        self._configs = data.get("sources", {})
        self._loaded = True
        logger.debug("Loaded %d source configs from %s", len(self._configs), self._config_path)

    def _instantiate(self, source_id: str) -> DataSourceBase:
        """Dynamically import adapter class and instantiate with config params."""
        cfg = self._configs[source_id]
        adapter_path: str = cfg["adapter"]      # e.g. "ingestion.futures.YahooFuturesSource"
        params: dict = cfg.get("params", {})
        enabled: bool = cfg.get("enabled", True)

        if not enabled:
            raise ValueError(f"Source '{source_id}' is disabled in sources.yaml")

        # Dynamic import
        module_path, class_name = adapter_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Inject cache_dir from settings
        cache_dir = str(settings.raw_dir / source_id)

        # Coerce enum strings if needed
        processed_params = _coerce_params(params)
        return cls(cache_dir=cache_dir, **processed_params)

    def get(self, source_id: str) -> DataSourceBase:
        """Return (cached) adapter instance for the given source_id."""
        self._load_config()
        if source_id not in self._configs:
            raise KeyError(f"Unknown source: '{source_id}'. Available: {self.list_sources()}")
        if source_id not in self._instances:
            self._instances[source_id] = self._instantiate(source_id)
        return self._instances[source_id]

    def list_sources(self, enabled_only: bool = False) -> list[str]:
        self._load_config()
        if enabled_only:
            return [k for k, v in self._configs.items() if v.get("enabled", True)]
        return list(self._configs.keys())

    def list_enabled(self) -> list[str]:
        return self.list_sources(enabled_only=True)


def _coerce_params(params: dict) -> dict:
    """Convert YAML string values to domain types where needed."""
    from contracts.schemas import CoffeeVariety, Exchange
    coerced = {}
    for k, v in params.items():
        if k == "variety" and isinstance(v, str):
            coerced[k] = CoffeeVariety(v)
        elif k == "exchange" and isinstance(v, str):
            coerced[k] = Exchange(v)
        else:
            coerced[k] = v
    return coerced


# Module-level singleton
source_registry = SourceRegistry()
