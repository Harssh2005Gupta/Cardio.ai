import pandas as pd

from src.explainability.shap_explainer import build_explainer, build_full_evidence, explain_instance
from src.models.inference import load_artifacts, predict_one

SAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0,
    "restecg": 1, "thalach": 145, "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
}


def _get_explainer(trained_models_dir):
    loaded = load_artifacts(trained_models_dir)
    background = pd.read_csv(trained_models_dir / "background_sample.csv")
    explainer = build_explainer(loaded.model, loaded.model_type, background)
    return loaded, explainer, background


def test_explain_instance_returns_one_contribution_per_feature(trained_models_dir):
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = explain_instance(explainer, instance, loaded.model_type)
    assert len(evidence.local_contributions) == len(loaded.feature_columns)
    assert evidence.output_space == "probability_space"


def test_contribution_directions_are_valid_labels(trained_models_dir):
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = explain_instance(explainer, instance, loaded.model_type)
    allowed = {"increases_model_output", "decreases_model_output", "minimal_contribution"}
    assert all(c.direction in allowed for c in evidence.local_contributions)


def test_top_contributions_sorted_by_absolute_impact(trained_models_dir):
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = explain_instance(explainer, instance, loaded.model_type)
    top = evidence.top(5)
    abs_values = [abs(c.contribution) for c in top]
    assert abs_values == sorted(abs_values, reverse=True)


def test_as_dict_has_structured_schema(trained_models_dir):
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = explain_instance(explainer, instance, loaded.model_type)
    d = evidence.top(1)[0].as_dict()
    assert set(d.keys()) == {"feature", "feature_label", "value", "contribution", "direction", "explanation"}


def test_top_positive_and_negative_are_disjoint(trained_models_dir):
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = explain_instance(explainer, instance, loaded.model_type)
    pos_features = {c.feature for c in evidence.top_positive(5)}
    neg_features = {c.feature for c in evidence.top_negative(5)}
    assert pos_features.isdisjoint(neg_features)


def test_build_full_evidence_includes_global_importance(trained_models_dir):
    loaded, explainer, background = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    evidence = build_full_evidence(explainer, instance, loaded.model_type, background)
    assert len(evidence.global_importance) == len(loaded.feature_columns)
    assert all(g.mean_abs_shap >= 0 for g in evidence.global_importance)
    # sorted descending
    values = [g.mean_abs_shap for g in evidence.global_importance]
    assert values == sorted(values, reverse=True)


def test_explainer_works_uniformly_regardless_of_calibration(trained_models_dir):
    """The uniform explainer must work whether the underlying model is a
    raw estimator or a CalibratedClassifierCV-wrapped one — this is exactly
    the case v1's TreeExplainer/LinearExplainer special-casing could not
    handle."""
    loaded, explainer, _ = _get_explainer(trained_models_dir)
    _, instance = predict_one(loaded, SAMPLE_PATIENT)
    # Must not raise regardless of loaded.metadata["calibration_applied"].
    evidence = explain_instance(explainer, instance, loaded.model_type)
    assert len(evidence.local_contributions) > 0
