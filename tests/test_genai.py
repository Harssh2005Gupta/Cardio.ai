import json
from unittest.mock import MagicMock, patch

from src.genai import explainer as genai_module

SAMPLE_FACTORS = [
    {"feature": "cp_2", "feature_label": "cp: Non-anginal pain", "value": 1.0,
     "contribution": 0.85, "direction": "increases_model_output",
     "explanation": "pushed up the model's predicted risk"},
    {"feature": "oldpeak", "feature_label": "ST depression (oldpeak)", "value": -0.5,
     "contribution": -0.4, "direction": "decreases_model_output",
     "explanation": "pushed down the model's predicted risk"},
]


def test_fallback_used_when_no_api_key(monkeypatch):
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", None, raising=False)
    result = genai_module.generate_explanation(0.72, "logistic_regression", SAMPLE_FACTORS)
    assert result.source == "fallback_template"
    assert "72.0%" in result.summary
    assert len(result.key_factors) > 0
    assert result.limitations  # never empty
    assert "not a medical diagnosis" in result.safety_note.lower()


def test_fallback_never_invents_facts_beyond_input():
    result = genai_module._fallback_explanation(0.5, SAMPLE_FACTORS)
    factor_texts = " ".join(kf.explanation for kf in result.key_factors)
    # the fallback only ever echoes the SHAP-supplied explanation strings
    assert factor_texts == " ".join(f["explanation"] for f in SAMPLE_FACTORS[:3])


def test_genai_call_mocked_success(monkeypatch):
    """Proves the OpenAI call path works without ever hitting the network."""
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    fake_payload = {
        "summary": "Mocked summary.",
        "key_factors": [{"feature": "cp_2", "direction": "increases_model_output", "explanation": "mocked"}],
        "limitations": ["mocked limitation"],
        "safety_note": "This is an educational model output and not a medical diagnosis.",
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(fake_payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "genai"
    assert result.summary == "Mocked summary."
    assert result.key_factors[0].explanation == "mocked"


def test_genai_call_failure_falls_back_safely(monkeypatch):
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("simulated network failure")

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "fallback_error"
    assert result.summary  # still produced a usable explanation, did not crash


def test_no_diagnosis_language_in_system_prompt():
    """Guardrail check: the system prompt must explicitly forbid diagnosis/treatment claims."""
    prompt = genai_module.SYSTEM_PROMPT.lower()
    assert "do not independently diagnose" in prompt
    assert "do not provide treatment" in prompt

def test_invalid_schema_triggers_fallback(monkeypatch):
    """Missing required field ('limitations') must be rejected by schema
    validation, not passed through to the user."""
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    bad_payload = {
        "summary": "Missing limitations and safety note fields.",
        "key_factors": [],
        # "limitations" and "safety_note" deliberately omitted
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(bad_payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "fallback_invalid_schema"
    assert result.summary  # still usable


def test_safety_note_missing_disclaimer_triggers_fallback(monkeypatch):
    """A schema-complete but non-compliant safety_note (missing the required
    disclaimer phrase) must also be rejected."""
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    bad_payload = {
        "summary": "A summary.",
        "key_factors": [{"feature": "cp_2", "direction": "increases_model_output", "explanation": "x"}],
        "limitations": ["some limitation"],
        "safety_note": "Everything is fine.",  # missing required disclaimer phrase
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(bad_payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "fallback_invalid_schema"


def test_safety_filter_catches_diagnostic_language(monkeypatch):
    """A schema-valid response that still contains forbidden diagnostic
    language must be caught by the keyword safety filter, not shown as-is."""
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    unsafe_payload = {
        "summary": "Based on this, you have coronary artery disease and should start treatment.",
        "key_factors": [{"feature": "cp_2", "direction": "increases_model_output", "explanation": "x"}],
        "limitations": ["some limitation"],
        "safety_note": "This is an educational model output and not a medical diagnosis.",
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(unsafe_payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "fallback_safety_filter"


def test_safe_response_passes_through_unmodified(monkeypatch):
    """Sanity check: a genuinely safe, schema-valid response is NOT
    incorrectly caught by the filter."""
    monkeypatch.setattr(genai_module.CONFIG, "openai_api_key", "sk-test-fake-key", raising=False)

    safe_payload = {
        "summary": "The model estimated a moderate probability based on a few contributing factors.",
        "key_factors": [{"feature": "cp_2", "direction": "increases_model_output", "explanation": "Chest pain type contributed positively."}],
        "limitations": ["Not clinically validated."],
        "safety_note": "This is an educational model output and not a medical diagnosis.",
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(safe_payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = genai_module.generate_explanation(0.6, "logistic_regression", SAMPLE_FACTORS)

    assert result.source == "genai"
