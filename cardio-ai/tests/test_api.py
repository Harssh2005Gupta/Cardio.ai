import pytest
from fastapi.testclient import TestClient

SAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0,
    "restecg": 1, "thalach": 145, "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
}


@pytest.fixture()
def client(trained_models_dir):
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_metadata_endpoint_has_cv_and_calibration_fields(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "best_model_type" in body
    assert "cv_results" in body and len(body["cv_results"]) == 5
    assert "test_metrics" in body
    assert "calibration_applied" in body
    assert "model_version" in body
    assert "selection_method" in body
    assert "random_seed" in body


def test_predict_endpoint_valid_input(client):
    r = client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert "calibrated" in body
    assert "disclaimer" in body
    # No risk-tier language anywhere in the response.
    assert "low" not in body["disclaimer"].lower()
    assert "moderate" not in body["disclaimer"].lower()
    assert "high" not in body["disclaimer"].lower()


def test_predict_endpoint_invalid_input_returns_422(client):
    bad = dict(SAMPLE_PATIENT)
    bad["sex"] = 99
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert "errors" in body
    assert body["errors"][0]["field"]


def test_predict_endpoint_missing_field_returns_422(client):
    bad = dict(SAMPLE_PATIENT)
    del bad["age"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_explain_endpoint_returns_structured_evidence(client):
    r = client.post("/explain", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_contributions"]) > 0
    assert body["output_space"] == "probability_space"
    assert "base_value" in body
    assert len(body["global_importance"]) > 0


def test_generate_explanation_endpoint_without_api_key(client):
    r = client.post("/generate-explanation", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in (
        "fallback_template", "fallback_error", "fallback_invalid_schema", "fallback_safety_filter", "genai"
    )
    assert "not a medical diagnosis" in body["safety_note"].lower()


def test_response_headers_present_on_every_request(client):
    """Sanity check that the structured-logging middleware doesn't break
    normal request/response handling."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/json")
