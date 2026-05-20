"""
experiment/tracker.py
=====================
Lightweight experiment tracking backed by JSON files.

Designed as a drop-in that can be swapped for MLflow later
(both implement ExperimentTrackerBase).

Storage layout:
  {experiments_dir}/
    {run_id}/
      meta.json        — run metadata, tags, start/end time
      params.json      — hyperparameters
      metrics.json     — scalar metrics
      artifacts/       — any files (model pickles, plots, etc.)

Usage:
    from experiment.tracker import FileExperimentTracker
    from config.settings import settings

    tracker = FileExperimentTracker(settings.experiments_dir)
    run_id = tracker.start_run("sarimax_arabica_v1", tags={"variety": "arabica"})
    tracker.log_params(run_id, model.get_params())
    tracker.log_metrics(run_id, {"directional_accuracy": 0.57, "brier": 0.23})
    tracker.end_run(run_id)
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from contracts.interfaces import ExperimentTrackerBase


class FileExperimentTracker(ExperimentTrackerBase):
    """File-backed experiment tracker. Thread-safe for single-process use."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        run_id = f"{run_name}_{datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        meta = {
            "run_id": run_id,
            "run_name": run_name,
            "tags": tags or {},
            "status": "running",
            "start_time": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "end_time": None,
        }
        self._write_json(run_dir / "meta.json", meta)
        self._write_json(run_dir / "params.json", {})
        self._write_json(run_dir / "metrics.json", {})
        return run_id

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        path = self._run_dir(run_id) / "params.json"
        existing = self._read_json(path)
        existing.update({k: self._serialise(v) for k, v in params.items()})
        self._write_json(path, existing)

    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        path = self._run_dir(run_id) / "metrics.json"
        existing = self._read_json(path)
        # Support step-based metrics: {metric: [val1, val2, ...]}
        for k, v in metrics.items():
            if k not in existing:
                existing[k] = []
            existing[k].append({"value": v, "ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()})
        self._write_json(path, existing)

    def log_artifact(self, run_id: str, path: str) -> None:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        dest = self._run_dir(run_id) / "artifacts" / src.name
        shutil.copy2(src, dest)

    def end_run(self, run_id: str) -> None:
        meta_path = self._run_dir(run_id) / "meta.json"
        meta = self._read_json(meta_path)
        meta["status"] = "finished"
        meta["end_time"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        self._write_json(meta_path, meta)

    # ------------------------------------------------------------------
    # Convenience / query
    # ------------------------------------------------------------------

    def list_runs(self, run_name_filter: Optional[str] = None) -> List[Dict]:
        runs = []
        for run_dir in sorted(self.base_dir.iterdir()):
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_json(meta_path)
            if run_name_filter and run_name_filter not in meta.get("run_name", ""):
                continue
            runs.append(meta)
        return runs

    def get_metrics(self, run_id: str) -> Dict[str, List[float]]:
        raw = self._read_json(self._run_dir(run_id) / "metrics.json")
        return {k: [entry["value"] for entry in v] for k, v in raw.items()}

    def get_latest_metric(self, run_id: str, metric: str) -> Optional[float]:
        metrics = self.get_metrics(run_id)
        values = metrics.get(metric, [])
        return values[-1] if values else None

    def get_params(self, run_id: str) -> Dict[str, Any]:
        return self._read_json(self._run_dir(run_id) / "params.json")

    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> "import pandas; pandas.DataFrame":
        import pandas as pd
        rows = []
        for rid in run_ids:
            meta = self._read_json(self._run_dir(rid) / "meta.json")
            row = {"run_id": rid, "run_name": meta.get("run_name", rid)}
            for m in metrics:
                v = self.get_latest_metric(rid, m)
                row[m] = round(v, 4) if v is not None else None
            row.update(self.get_params(rid))
            rows.append(row)
        return pd.DataFrame(rows).set_index("run_id")

    def best_run(self, metric: str, higher_is_better: bool = True) -> Optional[str]:
        """Return run_id with best value of a metric across all runs."""
        best_id, best_val = None, None
        for meta in self.list_runs():
            rid = meta["run_id"]
            v = self.get_latest_metric(rid, metric)
            if v is None:
                continue
            if best_val is None:
                best_id, best_val = rid, v
            elif (higher_is_better and v > best_val) or (not higher_is_better and v < best_val):
                best_id, best_val = rid, v
        return best_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _serialise(v: Any) -> Any:
        """Make value JSON-serialisable."""
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return list(v)
        if isinstance(v, dict):
            return v
        return str(v)
