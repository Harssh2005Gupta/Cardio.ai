import json

from src.models.model_registry import load_metadata


def test_training_produces_all_expected_artifacts(trained_models_dir):
    files = {p.name for p in trained_models_dir.iterdir()}
    expected = {
        "preprocessing_pipeline.pkl", "feature_columns.json", "metadata.json",
        "background_sample.csv",
    }
    assert expected.issubset(files)
    # exactly one of best_model.pkl / best_model.keras should exist
    assert ("best_model.pkl" in files) != ("best_model.keras" in files)


def test_metadata_has_all_five_models_cv_scored(trained_models_dir):
    metadata = load_metadata(trained_models_dir)
    cv_results = metadata["cv_results"]
    expected_models = {"logistic_regression", "decision_tree", "random_forest", "svm", "neural_network_dl"}
    assert expected_models.issubset(cv_results.keys())
    for model_name, scores in cv_results.items():
        assert 0.0 <= scores["mean_roc_auc"] <= 1.0
        assert scores["std_roc_auc"] >= 0.0
        assert len(scores["fold_scores"]) == metadata["cv_folds"]


def test_best_model_selected_by_cv_score_only(trained_models_dir):
    """Proves selection used cv_results, not test_metrics: the stored
    best_model_type must be the argmax of cv_results' mean_roc_auc."""
    metadata = load_metadata(trained_models_dir)
    cv_results = metadata["cv_results"]
    expected_best = max(cv_results, key=lambda k: cv_results[k]["mean_roc_auc"])
    assert metadata["best_model_type"] == expected_best


def test_selection_method_documents_test_set_exclusion(trained_models_dir):
    metadata = load_metadata(trained_models_dir)
    assert "never used for model selection" in metadata["selection_method"].lower()


def test_test_metrics_present_for_winner_only(trained_models_dir):
    metadata = load_metadata(trained_models_dir)
    test_metrics = metadata["test_metrics"]
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier_score"]:
        assert key in test_metrics
        assert isinstance(test_metrics[key], float)
    assert 0.0 <= test_metrics["roc_auc"] <= 1.0
    assert 0.0 <= test_metrics["brier_score"] <= 1.0


def test_calibration_info_present(trained_models_dir):
    metadata = load_metadata(trained_models_dir)
    assert "calibration_applied" in metadata
    assert "brier_before_calibration" in metadata
    assert "brier_after_calibration" in metadata
    if metadata["calibration_applied"]:
        assert metadata["brier_before_calibration"] is not None
        assert metadata["brier_after_calibration"] is not None
    else:
        # DL winner: calibration intentionally not applied (documented limitation).
        assert metadata["best_model_type"] == "neural_network_dl"


def test_metadata_has_reproducibility_fields(trained_models_dir):
    metadata = load_metadata(trained_models_dir)
    for field in ["dataset_hash", "trained_at_utc", "random_seed", "model_version", "cv_folds"]:
        assert metadata.get(field) is not None
    assert metadata["model_version"].startswith(metadata["best_model_type"])


def test_feature_columns_match_pipeline_output(trained_models_dir):
    with open(trained_models_dir / "feature_columns.json") as f:
        cols = json.load(f)
    assert len(cols) > 0
    assert "age" in cols
