"""
src/genai/explainer.py
-------------------------
Turns model prediction + SHAP evidence into a structured, patient-friendly
explanation via OpenAI. The GenAI layer is a communicator, not a predictor.

v2 adds two guardrails beyond v1's "never crash, always fall back":

  1. SCHEMA VALIDATION — the raw OpenAI JSON response is parsed into a
     Pydantic model. Missing/empty required fields, or a safety_note that
     doesn't contain the required disclaimer phrase, is treated as an
     invalid response and triggers the fallback (source="fallback_invalid_schema").

  2. SAFETY FILTER — even a schema-valid response is scanned for a small
     denylist of diagnostic/treatment language (e.g. "you have", "diagnosed
     with", "take <medication>") that the system prompt already forbids but
     which an LLM could still produce. A match triggers the fallback
     (source="fallback_safety_filter") rather than being shown to the user.

Both checks are defense-in-depth on top of the prompt-level instruction —
neither replaces it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.utils.config import CONFIG
from src.utils.logging_config import get_logger, redact

logger = get_logger(__name__)

REQUIRED_SAFETY_PHRASE = "not a medical diagnosis"

SYSTEM_PROMPT = """You are a clinical communication assistant for an educational prototype.
You explain a machine learning model's cardiovascular-risk output and its SHAP evidence in
plain language for a patient with no medical background.

Explain the supplied model output and SHAP evidence. Do not independently diagnose the
patient. Do not claim that a feature causes disease — SHAP shows association with the
model's output, not medical causation. Do not provide treatment or medication instructions.
Do not invent symptoms, medical history, test results, or facts not supplied to you.
Clearly state uncertainty and limitations.

Respond ONLY with a JSON object matching exactly this schema, no other text:
{
  "summary": "2-3 sentence plain-language summary of the result",
  "key_factors": [
    {"feature": "<feature name as given>", "direction": "<as given>", "explanation": "<1 sentence, plain language>"}
  ],
  "limitations": ["<short limitation statement>", "..."],
  "safety_note": "This is an educational model output and not a medical diagnosis."
}
"""

# Small, deliberately conservative denylist — false positives just mean an
# extra template fallback, which is safe; false negatives are the risk we're
# guarding against.
_UNSAFE_PATTERNS = [
    r"\byou have\b",
    r"\bdiagnos(e|ed|is|ing)\b",
    r"\byou (should|must) take\b",
    r"\bprescri(be|bed|ption)\b",
    r"\brecommend(ed)? (medication|drug|dose|dosage)\b",
    r"\byou are (suffering|at risk of dying)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


class KeyFactorSchema(BaseModel):
    feature: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class GenAIResponseSchema(BaseModel):
    """Validates the raw parsed JSON from OpenAI before it's trusted."""
    summary: str = Field(min_length=1)
    key_factors: List[KeyFactorSchema] = Field(default_factory=list)
    limitations: List[str] = Field(min_length=1)
    safety_note: str = Field(min_length=1)

    @field_validator("safety_note")
    @classmethod
    def safety_note_must_contain_disclaimer(cls, v: str) -> str:
        if REQUIRED_SAFETY_PHRASE not in v.lower():
            raise ValueError(f"safety_note missing required phrase: '{REQUIRED_SAFETY_PHRASE}'")
        return v


@dataclass
class KeyFactor:
    feature: str
    direction: str
    explanation: str


@dataclass
class GenAIExplanation:
    summary: str
    key_factors: List[KeyFactor] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    safety_note: str = "This is an educational model output and not a medical diagnosis."
    # "genai" | "fallback_template" | "fallback_error" | "fallback_invalid_schema" | "fallback_safety_filter"
    source: str = "genai"

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_factors": [kf.__dict__ for kf in self.key_factors],
            "limitations": self.limitations,
            "safety_note": self.safety_note,
            "source": self.source,
        }


STANDARD_LIMITATIONS = [
    "This model was trained on a small, publicly available research dataset and has not been clinically validated.",
    "SHAP explains the model's behavior, not medical causation.",
    "This output does not account for medical history, medications, or lab results not entered here.",
]


def _build_user_prompt(probability: float, model_name: str, top_factors: List[dict]) -> str:
    payload = {
        "model_predicted_probability": round(probability, 4),
        "dataset_positive_class_meaning": "target=1 in the training dataset (research-dataset label, not a diagnosis)",
        "model_name": model_name,
        "top_shap_factors": top_factors,
    }
    return "Model output and evidence:\n" + json.dumps(payload, indent=2)


def _fallback_explanation(probability: float, top_factors: List[dict], source: str = "fallback_template") -> GenAIExplanation:
    key_factors = [
        KeyFactor(feature=f["feature_label"], direction=f["direction"], explanation=f["explanation"])
        for f in top_factors[:3]
    ]
    summary = (
        f"The model estimated a {probability * 100:.1f}% probability for the positive class "
        f"in this dataset, based on the factors below."
    )
    return GenAIExplanation(summary=summary, key_factors=key_factors, limitations=STANDARD_LIMITATIONS, source=source)


def _validate_schema(raw_json: str) -> Optional[GenAIResponseSchema]:
    try:
        parsed = json.loads(raw_json)
        return GenAIResponseSchema.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(f"GenAI response failed schema validation: {exc.__class__.__name__}")
        return None


def _contains_unsafe_language(schema: GenAIResponseSchema) -> bool:
    text = schema.summary + " " + " ".join(kf.explanation for kf in schema.key_factors)
    return bool(_UNSAFE_RE.search(text))


def generate_explanation(probability: float, model_name: str, top_factors: List[dict]) -> GenAIExplanation:
    """top_factors: list of dicts as produced by
    EvidenceBundle.top()[i].as_dict() (feature_label/direction/explanation present)."""
    api_key = CONFIG.openai_api_key
    if not api_key:
        logger.info("No OPENAI_API_KEY configured; returning template fallback explanation.")
        return _fallback_explanation(probability, top_factors)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=CONFIG.genai_timeout_seconds)
        user_prompt = _build_user_prompt(probability, model_name, top_factors)

        response = client.chat.completions.create(
            model=CONFIG.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

        schema = _validate_schema(raw)
        if schema is None:
            return _fallback_explanation(probability, top_factors, source="fallback_invalid_schema")

        if _contains_unsafe_language(schema):
            logger.warning("GenAI response matched the safety denylist; using fallback explanation.")
            return _fallback_explanation(probability, top_factors, source="fallback_safety_filter")

        key_factors = [KeyFactor(feature=kf.feature, direction=kf.direction, explanation=kf.explanation) for kf in schema.key_factors]
        return GenAIExplanation(
            summary=schema.summary,
            key_factors=key_factors,
            limitations=schema.limitations,
            safety_note=schema.safety_note,
            source="genai",
        )

    except Exception as exc:  # noqa: BLE001 — any failure must degrade safely, never crash the app
        logger.warning(
            f"GenAI call failed ({exc.__class__.__name__}); using fallback explanation. "
            f"(key={redact(api_key)})"
        )
        return _fallback_explanation(probability, top_factors, source="fallback_error")
