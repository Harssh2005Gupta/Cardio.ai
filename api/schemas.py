"""
api/schemas.py
-----------------
Pydantic request/response models for the FastAPI backend. Field-level
constraints mirror src/data/validation.py's allowed ranges/categories, so
malformed patient data is rejected at the API boundary with a clear 422
before it ever reaches the model.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289,
                "fbs": 0, "restecg": 1, "thalach": 145, "exang": 1,
                "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
            }
        }
    )

    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0, le=300)
    chol: float = Field(..., ge=0, le=800)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, le=260)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=-3, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=1, le=3)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    calibrated: bool
    dataset_positive_class: str = "target=1 in the supplied training dataset"
    disclaimer: str = "Educational prototype only. Not a medical device or diagnosis. This is a model-predicted probability, not a clinical risk category."


class FeatureContributionOut(BaseModel):
    feature: str
    feature_label: str
    value: float
    contribution: float
    direction: str
    explanation: str


class GlobalFeatureImportanceOut(BaseModel):
    feature: str
    feature_label: str
    mean_abs_shap: float


class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    output_space: str
    base_value: float
    top_contributions: List[FeatureContributionOut]
    top_positive: List[FeatureContributionOut]
    top_negative: List[FeatureContributionOut]
    global_importance: List[GlobalFeatureImportanceOut]
    limitation_note: str = "SHAP explains the model's behavior; it does not establish medical causation."


class KeyFactorOut(BaseModel):
    feature: str
    direction: str
    explanation: str


class GenerateExplanationResponse(BaseModel):
    summary: str
    key_factors: List[KeyFactorOut]
    limitations: List[str]
    safety_note: str
    # "genai" | "fallback_template" | "fallback_error" | "fallback_invalid_schema" | "fallback_safety_filter"
    source: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    best_model_type: str
    model_version: Optional[str] = None
    selection_method: Optional[str] = None
    cv_folds: Optional[int] = None
    cv_results: dict = Field(default_factory=dict)
    test_metrics: dict = Field(default_factory=dict)
    calibration_applied: Optional[bool] = None
    brier_before_calibration: Optional[float] = None
    brier_after_calibration: Optional[float] = None
    trained_at_utc: Optional[str] = None
    dataset_hash: Optional[str] = None
    n_training_rows: Optional[int] = None
    n_features: Optional[int] = None
    random_seed: Optional[int] = None
