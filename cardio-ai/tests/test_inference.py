from src.models.inference import load_artifacts, predict_one

SAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0,
    "restecg": 1, "thalach": 145, "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
}


def test_load_artifacts(trained_models_dir):
    loaded = load_artifacts(trained_models_dir)
    assert loaded.model is not None
    assert loaded.model_type in {
        "logistic_regression", "decision_tree", "random_forest", "svm", "neural_network_dl"
    }
    assert len(loaded.feature_columns) > 0


def test_predict_one_returns_valid_probability(trained_models_dir):
    loaded = load_artifacts(trained_models_dir)
    proba, instance = predict_one(loaded, SAMPLE_PATIENT)
    assert 0.0 <= proba <= 1.0
    assert instance.shape[0] == 1
    assert list(instance.columns) == loaded.feature_columns


def test_predict_one_is_deterministic(trained_models_dir):
    loaded = load_artifacts(trained_models_dir)
    proba1, _ = predict_one(loaded, SAMPLE_PATIENT)
    proba2, _ = predict_one(loaded, SAMPLE_PATIENT)
    assert proba1 == proba2
