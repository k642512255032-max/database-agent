"""Model bundles and the registry of already-trained models.

A bundle = estimator + fitted AutoFeatureEngineer + metadata, saved with joblib.
NOTE: joblib files are pickles - only load bundles you created yourself.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib

from agent.config import settings

SUPERVISED = {"classification", "regression"}
TASKS = {"classification", "regression", "clustering", "dimensionality_reduction", "anomaly_detection"}


@dataclass
class ModelBundle:
    name: str
    algorithm: str                  # decision_tree | random_forest | knn | kmeans | pca | isolation_forest | dbscan
    task: str                       # one of TASKS
    description: str
    base_sql: str                   # query that produced the training data
    feature_columns: list[str]      # raw columns the model needs
    estimator: Any
    feature_engineer: Any
    target: Optional[str] = None
    id_column: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    extra: dict = field(default_factory=dict)   # e.g. KNN training ids/labels, DBSCAN core points

    def card(self) -> dict:
        """JSON-safe summary (what the LLM and the UI see)."""
        return {
            "name": self.name, "algorithm": self.algorithm, "task": self.task,
            "description": self.description, "target": self.target,
            "id_column": self.id_column, "feature_columns": self.feature_columns,
            "base_sql": self.base_sql, "metrics": self.metrics, "trained_at": self.trained_at,
        }


class ModelRegistry:
    def __init__(self, models_dir: Path | None = None):
        self.dir = Path(models_dir or settings.models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ModelBundle] = {}

    def save(self, bundle: ModelBundle) -> Path:
        path = self.dir / f"{bundle.name}.joblib"
        joblib.dump(bundle, path)
        manifest = self._read_manifest()
        manifest[bundle.name] = bundle.card()
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return path

    def remove(self, name: str) -> None:
        (self.dir / f"{name}.joblib").unlink(missing_ok=True)
        manifest = self._read_manifest()
        manifest.pop(name, None)
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        self._cache.pop(name, None)

    def cards(self) -> list[dict]:
        return list(self._read_manifest().values())

    def names(self) -> list[str]:
        return [c["name"] for c in self.cards()]

    def get(self, name: str) -> ModelBundle:
        if name not in self._cache:
            path = self.dir / f"{name}.joblib"
            if not path.exists():
                raise KeyError(f"No trained model named '{name}'. Available: {self.names()}")
            self._cache[name] = joblib.load(path)
        return self._cache[name]

    def manifest_text(self) -> str:
        cards = self.cards()
        if not cards:
            return "(no trained models available)"
        lines = []
        for c in cards:
            tgt = f", target={c['target']}" if c.get("target") else ""
            lines.append(f"- {c['name']}: {c['algorithm']} ({c['task']}{tgt}) - {c['description']}")
        return "\n".join(lines)

    def _read_manifest(self) -> dict:
        p = self.dir / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}
