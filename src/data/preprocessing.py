"""
src/data/preprocessing.py
----------------------------
Leak-free preprocessing built as a single scikit-learn Pipeline, so the
*exact* fitted transformer (imputers, encoder, scaler) used at training time
is serialized once and reused, unchanged, for every later prediction — by
train.py, api/main.py, and app/streamlit_app.py alike.

Critical rule enforced here: fit_pipeline() must only ever be called on the
training split. validate/transform on val/test/inference data must always
go through pipeline.transform(), never pipeline.fit_transform().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target"
NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
RAW_FEATURE_ORDER = NUMERICAL_COLS + CATEGORICAL_COLS

CATEGORY_LABELS = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {1: "Fasting blood sugar > 120 mg/dl", 0: "Fasting blood sugar <= 120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable/definite LV hypertrophy"},
    "exang": {1: "Yes", 0: "No"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels", 4: "4 vessels"},
    "thal": {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
}


def build_pipeline() -> ColumnTransformer:
    """An unfitted preprocessing pipeline. Fit ONLY on the training split."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERICAL_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def get_output_feature_names(pipeline: ColumnTransformer) -> List[str]:
    """Feature names after transform, in the exact order transform() emits them."""
    cat_encoder = pipeline.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    return NUMERICAL_COLS + cat_names


def transform_to_df(pipeline: ColumnTransformer, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply an ALREADY-FITTED pipeline and return a named dataframe."""
    arr = pipeline.transform(X_raw[RAW_FEATURE_ORDER])
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    cols = get_output_feature_names(pipeline)
    return pd.DataFrame(arr, columns=cols, index=X_raw.index)


def save_pipeline(pipeline: ColumnTransformer, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "preprocessing_pipeline.pkl")
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(get_output_feature_names(pipeline), f, indent=2)


def load_pipeline(model_dir: Path):
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / "preprocessing_pipeline.pkl")
    with open(model_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return pipeline, feature_columns


def display_name(feature: str) -> str:
    """Human-readable label for a transformed feature name, for UI/GenAI use."""
    friendly = {
        "age": "Age", "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol", "thalach": "Max heart rate achieved",
        "oldpeak": "ST depression (oldpeak)",
    }
    if feature in friendly:
        return friendly[feature]
    for cat_col, labels in CATEGORY_LABELS.items():
        prefix = f"{cat_col}_"
        if feature.startswith(prefix):
            try:
                code = int(feature[len(prefix):])
                return f"{cat_col}: {labels.get(code, code)}"
            except ValueError:
                return feature
    return feature
