"""
src/explainability/shap_explainer.py
---------------------------------------
Structured SHAP evidence layer.

Design change from v1: v1 special-cased shap.TreeExplainer / LinearExplainer
per model type for speed/exactness. That assumption breaks once the winning
classical model is wrapped in CalibratedClassifierCV (train.py v2) — a
calibrated ensemble is no longer a raw tree or a raw linear model, so a
TreeExplainer/LinearExplainer would either error or explain the wrong thing.

v2 always uses a single, uniform, model-agnostic shap.Explainer wrapped
around predict_proba (or, for the Keras DL model, .predict). This is
slightly slower than an exact TreeExplainer but is correct for every model
type this project can produce, including the calibrated ensemble — and for
a single-patient explanation the speed difference is not noticeable.

Output space: every path here explains predict_proba's positive-class
output directly, so all contributions are in PROBABILITY SPACE, consistently
across model types. This is stated explicitly so a contribution is never
mixed with mismatched units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import shap

from src.data.preprocessing import display_name

SHAP_OUTPUT_SPACE = "probability_space"
_MINIMAL_THRESHOLD = 0.01  # |contribution| below this is labeled "minimal"


def _direction(value: float) -> str:
    if abs(value) < _MINIMAL_THRESHOLD:
        return "minimal_contribution"
    return "increases_model_output" if value > 0 else "decreases_model_output"


@dataclass
class FeatureContribution:
    feature: str
    feature_label: str
    value: float
    contribution: float
    direction: str

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "feature_label": self.feature_label,
            "value": round(float(self.value), 4),
            "contribution": round(float(self.contribution), 4),
            "direction": self.direction,
            "explanation": (
                f"'{self.feature_label}' {self._verb()} the model's predicted probability "
                f"(SHAP contribution {self.contribution:+.3f}, {SHAP_OUTPUT_SPACE})."
            ),
        }

    def _verb(self) -> str:
        return {
            "increases_model_output": "pushed up",
            "decreases_model_output": "pushed down",
            "minimal_contribution": "had minimal effect on",
        }[self.direction]


@dataclass
class GlobalFeatureImportance:
    feature: str
    feature_label: str
    mean_abs_shap: float

    def as_dict(self) -> dict:
        return {"feature": self.feature, "feature_label": self.feature_label, "mean_abs_shap": round(self.mean_abs_shap, 4)}


@dataclass
class EvidenceBundle:
    """The complete structured SHAP evidence for one prediction: local
    per-feature contributions plus a global-importance snapshot, bundled so
    the API/UI/GenAI layer always receive one consistent object."""

    base_value: float
    output_space: str
    local_contributions: List[FeatureContribution] = field(default_factory=list)
    global_importance: List[GlobalFeatureImportance] = field(default_factory=list)

    def top(self, n: int = 5) -> List[FeatureContribution]:
        return sorted(self.local_contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]

    def top_positive(self, n: int = 5) -> List[FeatureContribution]:
        pos = [c for c in self.local_contributions if c.contribution > 0]
        return sorted(pos, key=lambda c: c.contribution, reverse=True)[:n]

    def top_negative(self, n: int = 5) -> List[FeatureContribution]:
        neg = [c for c in self.local_contributions if c.contribution < 0]
        return sorted(neg, key=lambda c: c.contribution)[:n]


def build_explainer(model, model_type: str, background: pd.DataFrame):
    """Uniform model-agnostic explainer. Works for calibrated classical
    models, raw classical models, and the Keras DL model alike."""

    def predict_fn(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=background.columns)
        if model_type == "neural_network_dl":
            return model.predict(x_df.values, verbose=0).ravel()
        return model.predict_proba(x_df)[:, 1]

    background_summary = shap.sample(background, min(50, len(background)), random_state=42)
    return shap.Explainer(predict_fn, background_summary)


def explain_instance(explainer, instance: pd.DataFrame, model_type: str) -> EvidenceBundle:
    """Local explanation for one patient row (already-transformed features)."""
    result = explainer(instance)
    values = np.array(result.values)[0]
    base_value = float(np.array(result.base_values).ravel()[0])

    contributions = [
        FeatureContribution(
            feature=col,
            feature_label=display_name(col),
            value=float(instance.iloc[0][col]),
            contribution=float(val),
            direction=_direction(float(val)),
        )
        for col, val in zip(instance.columns, values)
    ]
    return EvidenceBundle(base_value=base_value, output_space=SHAP_OUTPUT_SPACE, local_contributions=contributions)


_GLOBAL_IMPORTANCE_CACHE: dict = {}


def add_global_importance(evidence: EvidenceBundle, explainer, background: pd.DataFrame, sample_size: int = 15) -> EvidenceBundle:
    """Attaches a global mean(|SHAP|) importance ranking (over a background
    sample) to an existing EvidenceBundle."""
    cache_key = id(explainer)
    if cache_key in _GLOBAL_IMPORTANCE_CACHE:
        evidence.global_importance = list(_GLOBAL_IMPORTANCE_CACHE[cache_key])
        return evidence

    sample = background.sample(min(sample_size, len(background)), random_state=42)
    values = np.array(explainer(sample).values)
    mean_abs = np.abs(values).mean(axis=0)
    ranked = sorted(zip(sample.columns, mean_abs), key=lambda t: t[1], reverse=True)
    importance_list = [
        GlobalFeatureImportance(feature=f, feature_label=display_name(f), mean_abs_shap=float(v)) for f, v in ranked
    ]
    _GLOBAL_IMPORTANCE_CACHE[cache_key] = importance_list
    evidence.global_importance = list(importance_list)
    return evidence


def build_full_evidence(explainer, instance: pd.DataFrame, model_type: str, background: pd.DataFrame) -> EvidenceBundle:
    """Convenience: local + global evidence in one call — this is the
    structured evidence object the API and Streamlit UI should use."""
    evidence = explain_instance(explainer, instance, model_type)
    return add_global_importance(evidence, explainer, background)
