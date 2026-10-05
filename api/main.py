"""
api/main.py
-------------
FastAPI backend. Model + preprocessing pipeline are loaded ONCE at startup
(see the lifespan handler), never per-request.

Structured logging: every request logs method, path, status code, and
latency as structured fields — never the request body, so patient data
never lands in logs.

Run:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    ExplainResponse,
    FeatureContributionOut,
    GenerateExplanationResponse,
    GlobalFeatureImportanceOut,
    HealthResponse,
    KeyFactorOut,
    MetadataResponse,
    PatientRecord,
    PredictResponse,
)
from src.explainability.shap_explainer import build_explainer, build_full_evidence
from src.genai.explainer import generate_explanation
from src.models.inference import load_artifacts, predict_one
from src.utils.config import CONFIG
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

STATE: dict = {"loaded": None, "explainer": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(CONFIG.models_dir)
    if (models_dir / "metadata.json").exists():
        loaded = load_artifacts(models_dir)
        import pandas as pd
        background = pd.read_csv(models_dir / "background_sample.csv")
        STATE["loaded"] = loaded
        STATE["explainer"] = build_explainer(loaded.model, loaded.model_type, background)
        STATE["background"] = background
        from src.explainability.shap_explainer import EvidenceBundle, SHAP_OUTPUT_SPACE, add_global_importance
        dummy = EvidenceBundle(base_value=0.5, output_space=SHAP_OUTPUT_SPACE)
        add_global_importance(dummy, STATE["explainer"], background)
        logger.info(f"Loaded model '{loaded.model_type}' (version={loaded.metadata.get('model_version')}) and warmed explainer at startup.")
    else:
        logger.warning(
            "No trained model found in models/. Run `python -m src.models.train --data <csv>` first. "
            "API will return 503 until a model is available."
        )
    yield
    STATE.clear()


app = FastAPI(
    title="Cardio AI API",
    description="Educational prototype — not a medical device. See /health and /metadata.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    # Structured fields only — request/response bodies (which may contain
    # patient data) are never logged.
    logger.info(
        f"method={request.method} path={request.url.path} "
        f"status={response.status_code} latency_ms={latency_ms}"
    )
    if "application/json" in response.headers.get("content-type", "") and "charset" not in response.headers.get("content-type", ""):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Return field-level errors without echoing the raw invalid payload back.
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    logger.warning(f"422 validation error on {request.url.path}: {len(errors)} field(s) invalid")
    return JSONResponse(status_code=422, content={"detail": "Invalid patient data.", "errors": errors})


def _require_model():
    if STATE.get("loaded") is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run training first (see README) and restart the API.",
        )
    return STATE["loaded"], STATE["explainer"]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=STATE.get("loaded") is not None)


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    loaded, _ = _require_model()
    m = loaded.metadata
    return MetadataResponse(
        best_model_type=m["best_model_type"],
        model_version=m.get("model_version"),
        selection_method=m.get("selection_method"),
        cv_folds=m.get("cv_folds"),
        cv_results=m.get("cv_results", {}),
        test_metrics=m.get("test_metrics", {}),
        calibration_applied=m.get("calibration_applied"),
        brier_before_calibration=m.get("brier_before_calibration"),
        brier_after_calibration=m.get("brier_after_calibration"),
        trained_at_utc=m.get("trained_at_utc"),
        dataset_hash=m.get("dataset_hash"),
        n_training_rows=m.get("n_training_rows"),
        n_features=m.get("n_features"),
        random_seed=m.get("random_seed"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(record: PatientRecord):
    loaded, _ = _require_model()
    proba, _ = predict_one(loaded, record.model_dump())
    return PredictResponse(
        prediction=int(proba >= 0.5),
        probability=round(proba, 4),
        model_name=loaded.model_type,
        calibrated=bool(loaded.metadata.get("calibration_applied", False)),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(record: PatientRecord):
    loaded, explainer = _require_model()
    proba, instance = predict_one(loaded, record.model_dump())
    evidence = build_full_evidence(explainer, instance, loaded.model_type, STATE["background"])

    return ExplainResponse(
        prediction=int(proba >= 0.5),
        probability=round(proba, 4),
        model_name=loaded.model_type,
        output_space=evidence.output_space,
        base_value=round(evidence.base_value, 4),
        top_contributions=[FeatureContributionOut(**c.as_dict()) for c in evidence.top(8)],
        top_positive=[FeatureContributionOut(**c.as_dict()) for c in evidence.top_positive(5)],
        top_negative=[FeatureContributionOut(**c.as_dict()) for c in evidence.top_negative(5)],
        global_importance=[GlobalFeatureImportanceOut(**g.as_dict()) for g in evidence.global_importance[:8]],
    )


@app.post("/generate-explanation", response_model=GenerateExplanationResponse)
def generate_explanation_endpoint(record: PatientRecord):
    loaded, explainer = _require_model()
    proba, instance = predict_one(loaded, record.model_dump())
    evidence = build_full_evidence(explainer, instance, loaded.model_type, STATE["background"])
    top_factors = [c.as_dict() for c in evidence.top(5)]

    genai_result = generate_explanation(proba, loaded.model_type, top_factors)
    return GenerateExplanationResponse(
        summary=genai_result.summary,
        key_factors=[KeyFactorOut(**kf.__dict__) for kf in genai_result.key_factors],
        limitations=genai_result.limitations,
        safety_note=genai_result.safety_note,
        source=genai_result.source,
    )
