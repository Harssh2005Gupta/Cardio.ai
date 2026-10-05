"""
src/models/inference.py
--------------------------
Loads the trained model + fitted preprocessing pipeline once (via
load_artifacts, meant to be cached by the caller — FastAPI does this at
startup, Streamlit via st.cache_resource) and exposes a single predict()
function used identically by both the API and the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from src.data.preprocessing import RAW_FEATURE_ORDER, transform_to_df
from src.models.model_registry import load_metadata


class LoadedModel:
    def __init__(self, model, model_type: str, pipeline, feature_columns, metadata: dict):
        self.model = model
        self.model_type = model_type
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.metadata = metadata


def load_artifacts(models_dir: Path) -> LoadedModel:
    models_dir = Path(models_dir)
    metadata = load_metadata(models_dir)
    model_type = metadata["best_model_type"]

    if model_type == "neural_network_dl":
        import tensorflow as tf
        model = tf.keras.models.load_model(models_dir / "best_model.keras")
    else:
        model = joblib.load(models_dir / "best_model.pkl")

    pipeline = joblib.load(models_dir / "preprocessing_pipeline.pkl")
    import json
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    return LoadedModel(model, model_type, pipeline, feature_columns, metadata)


def predict_one(loaded: LoadedModel, raw_record: dict) -> Tuple[float, pd.DataFrame]:
    """Returns (probability_of_positive_class, transformed_feature_row)."""
    row = {col: raw_record[col] for col in RAW_FEATURE_ORDER}
    X_raw = pd.DataFrame([row])
    X_processed = transform_to_df(loaded.pipeline, X_raw)

    if loaded.model_type == "neural_network_dl":
        proba = float(loaded.model.predict(X_processed.values, verbose=0).ravel()[0])
    else:
        proba = float(loaded.model.predict_proba(X_processed)[:, 1][0])

    return proba, X_processed
