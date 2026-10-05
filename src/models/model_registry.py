"""
src/models/model_registry.py
------------------------------
Lightweight model registry: saves/loads a metadata.json describing which
model won, its hyperparameters, dataset fingerprint, training timestamp,
and metrics — the minimum needed for reproducibility without pulling in a
full MLOps platform.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic fingerprint of the training dataset's content."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def build_metadata(
    best_model_type: str,
    cv_results: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, float],
    dataset_fingerprint: str,
    n_rows: int,
    n_features: int,
    random_seed: int,
    cv_folds: int,
    selection_method: str,
    calibration_applied: bool,
    brier_before: float | None,
    brier_after: float | None,
) -> Dict[str, Any]:
    trained_at = datetime.now(timezone.utc).isoformat()
    return {
        "best_model_type": best_model_type,
        # Model selection is decided ONLY from cv_results (never test_metrics).
        "selection_method": selection_method,
        "cv_folds": cv_folds,
        "cv_results": cv_results,
        # test_metrics is computed exactly once, only for the already-selected
        # winner, after selection was locked in by CV.
        "test_metrics": test_metrics,
        "calibration_applied": calibration_applied,
        "brier_before_calibration": brier_before,
        "brier_after_calibration": brier_after,
        "dataset_hash": dataset_fingerprint,
        "n_training_rows": n_rows,
        "n_features": n_features,
        "random_seed": random_seed,
        "trained_at_utc": trained_at,
        "model_version": f"{best_model_type}-{dataset_fingerprint}-{trained_at[:10]}",
    }


def save_metadata(metadata: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(model_dir: Path) -> Dict[str, Any]:
    with open(Path(model_dir) / "metadata.json") as f:
        return json.load(f)
