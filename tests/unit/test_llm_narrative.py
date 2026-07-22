"""Tests for the LLM narrative generator (Phase 5 GenAI Credit Explanation Layer)."""

from __future__ import annotations

import json
from typing import Any

from ml.explain.llm_narrative import (
    LLMNarrativeOutput,
    NarrativeRequest,
    _build_user_prompt,
    _detect_refusal,
    _fallback_narrative,
    _parse_llm_response,
    _strip_pii,
    generate_narrative,
)


def _sample_factors() -> list[dict[str, str]]:
    return [
        {"name": "Risk Score", "value": "82.50", "status": "computed"},
        {"name": "Credit Component", "value": "0.85/0.40", "status": "live"},
        {"name": "DTI Component", "value": "0.78/0.25", "status": "computed"},
        {"name": "GST Compliance", "value": "non_compliant", "status": "non_compliant"},
    ]


def _sample_shap() -> list[dict[str, Any]]:
    return [
        {"feature_name": "Credit Score Recent Delta", "raw_value": "15", "shap_contribution": -1.70156, "direction": "decrease_default_risk"},
        {"feature_name": "Interest Rate Pct", "raw_value": "11.5", "shap_contribution": 0.42345, "direction": "increase_default_risk"},
        {"feature_name": "Debt-To-Income Ratio", "raw_value": "22.0", "shap_contribution": 0.31234, "direction": "increase_default_risk"},
    ]


def _sample_profile() -> dict[str, Any]:
    return {
        "name": "Jane Doe",
        "pan": "ABCDE1234F",
        "monthly_income": 120000,
        "existing_emis": 25000,
        "loan_amount": 500000,
        "tenure_months": 36,
        "home_ownership": "MORTGAGE",
    }


def _make_request(
    decision: str = "APPROVE",
    risk_score: float = 82.5,
    confidence: float = 0.95,
    **overrides: Any,
) -> NarrativeRequest:
    kwargs = dict(
        decision=decision,
        risk_score=risk_score,
        confidence=confidence,
        data_reliability=1.0,
        rule_version="RULE_SET_V2",
        model_version="XGB_V1",
        factors=_sample_factors(),
        shap_contributions=_sample_shap(),
        applicant_profile=_sample_profile(),
    )
    kwargs.update(overrides)
    return NarrativeRequest(**kwargs)


# --- PII Stripping ---


def test_strip_pii_removes_name_and_pan() -> None:
    safe = _strip_pii(_sample_profile())
    assert safe["name"] == "[REDACTED]"
    assert safe["pan"] == "[REDACTED]"
    assert safe["monthly_income"] == 120000  # non-PII preserved


def test_strip_pii_redacts_pan_like_strings() -> None:
    profile = {"pan": "ABCDE1234F"}
    safe = _strip_pii(profile)
    assert safe["pan"] == "[REDACTED]"


def test_strip_pii_redacts_phone_numbers() -> None:
    profile = {"phone": "9876543210"}
    safe = _strip_pii(profile)
    assert safe["phone"] == "[REDACTED]"


def test_strip_pii_keeps_non_pii_fields() -> None:
    profile = {"monthly_income": 120000, "loan_amount": 500000}
    safe = _strip_pii(profile)
    assert safe["monthly_income"] == 120000
    assert safe["loan_amount"] == 500000


# --- NarrativeRequest ---


def test_to_prompt_context_strips_pii() -> None:
    request = _make_request()
    context = request.to_prompt_context()
    assert context["applicant"]["name"] == "[REDACTED]"
    assert context["applicant"]["pan"] == "[REDACTED]"
    assert context["decision"] == "APPROVE"
    assert context["risk_score"] == 82.5


def test_to_prompt_context_includes_top_factors() -> None:
    request = _make_request()
    context = request.to_prompt_context()
    assert len(context["top_factors"]) == 4
    assert context["top_factors"][0]["factor"] == "Risk Score"


def test_to_prompt_context_includes_shap_factors() -> None:
    request = _make_request()
    context = request.to_prompt_context()
    assert len(context["shap_factors"]) == 3
    assert context["shap_factors"][0]["feature"] == "Credit Score Recent Delta"
    assert context["shap_factors"][0]["impact"] == -1.7016


# --- _build_user_prompt ---


def test_build_user_prompt_contains_decision_and_risk() -> None:
    request = _make_request()
    context = request.to_prompt_context()
    prompt = _build_user_prompt(context)
    assert "Decision: APPROVE" in prompt
    assert "Risk Score: 82.5" in prompt
    assert "Key Factors" in prompt
    assert "Model Feature Contributions" in prompt
    assert "Jane Doe" not in prompt
    assert "ABCDE1234F" not in prompt


def test_build_user_prompt_for_decline() -> None:
    request = _make_request(decision="DECLINE", risk_score=35.2)
    context = request.to_prompt_context()
    prompt = _build_user_prompt(context)
    assert "Decision: DECLINE" in prompt
    assert "Risk Score: 35.2" in prompt


# --- _parse_llm_response ---


def test_parse_valid_json_response() -> None:
    content = json.dumps({"narrative": "Your application was approved.", "advice": None})
    result = _parse_llm_response(content, fallback_used=False)
    assert result.narrative == "Your application was approved."
    assert result.advice is None
    assert not result.fallback_used
    assert not result.refusal


def test_parse_response_with_advice() -> None:
    content = json.dumps({
        "narrative": "Your application was declined.",
        "advice": "Reduce your DTI to 30%.",
    })
    result = _parse_llm_response(content, fallback_used=False)
    assert result.narrative == "Your application was declined."
    assert result.advice == "Reduce your DTI to 30%."


def test_parse_response_with_code_block() -> None:
    content = "```json\n{\"narrative\": \"Approved!\"}\n```"
    result = _parse_llm_response(content, fallback_used=False)
    assert result.narrative == "Approved!"


def test_parse_empty_content_returns_fallback() -> None:
    result = _parse_llm_response("", fallback_used=False)
    assert result.fallback_used


def test_parse_invalid_json_returns_fallback() -> None:
    result = _parse_llm_response("not valid json", fallback_used=False)
    assert result.fallback_used


# --- _detect_refusal ---


def test_detect_refusal_patterns() -> None:
    assert _detect_refusal("I am unable to generate that explanation")
    assert _detect_refusal("I cannot provide this information")
    assert _detect_refusal("As an AI, I cannot")
    assert not _detect_refusal("Your application was approved")


# --- _fallback_narrative ---


def test_fallback_approve_narrative() -> None:
    request = _make_request(decision="APPROVE", risk_score=82.5, confidence=0.95)
    result = _fallback_narrative(request)
    assert result.fallback_used
    assert "approved" in result.narrative.lower()
    assert "82.5" in result.narrative
    assert result.advice is None


def test_fallback_decline_narrative() -> None:
    request = _make_request(decision="DECLINE", risk_score=35.2, confidence=0.45)
    result = _fallback_narrative(request)
    assert result.fallback_used
    assert "declined" in result.narrative.lower()
    assert "35.2" in result.narrative
    assert result.advice is not None
    assert "debt" in result.advice.lower() or "dti" in result.advice.lower()


def test_fallback_review_narrative() -> None:
    request = _make_request(decision="NEEDS_REVIEW", risk_score=55.0, confidence=0.45)
    result = _fallback_narrative(request)
    assert result.fallback_used
    assert "review" in result.narrative.lower()
    assert "55.0" in result.narrative


def test_fallback_no_request() -> None:
    result = _fallback_narrative()
    assert result.fallback_used
    assert "unable to provide" in result.narrative.lower()


# --- generate_narrative (without LLM — should fallback) ---


def test_generate_narrative_falls_back_when_no_llm() -> None:
    request = _make_request(decision="APPROVE", risk_score=82.5)
    result = generate_narrative(request, llm_endpoint=None)
    assert result.fallback_used
    assert result.narrative
    assert "approved" in result.narrative.lower()


def test_generate_narrative_falls_back_on_bad_endpoint() -> None:
    request = _make_request(decision="DECLINE", risk_score=35.2)
    result = generate_narrative(request, llm_endpoint="http://localhost:19999/nonexistent")
    assert result.fallback_used


# --- Integration safety: no PII in prompt ---


def test_prompt_contains_no_raw_pii() -> None:
    request = _make_request()
    context = request.to_prompt_context()
    prompt = _build_user_prompt(context)
    assert "Jane Doe" not in prompt
    assert "ABCDE1234F" not in prompt
    assert "[REDACTED]" in prompt  # PII replaced with [REDACTED] placeholder
    assert "monthly_income" in prompt or "Monthly Income" in prompt
    assert "120000" in prompt  # non-PII values preserved


# --- Edge cases ---


def test_narrative_request_empty_factors() -> None:
    request = _make_request(factors=[], shap_contributions=[])
    context = request.to_prompt_context()
    assert context["top_factors"] == []
    assert context["shap_factors"] == []


def test_narrative_request_minimal_profile() -> None:
    request = _make_request(applicant_profile={"monthly_income": 50000})
    context = request.to_prompt_context()
    assert context["applicant"]["monthly_income"] == 50000


def test_parse_response_trims_whitespace() -> None:
    content = '  {"narrative": "  Trimmed.  ", "advice": null}  '
    result = _parse_llm_response(content, fallback_used=False)
    assert result.narrative == "Trimmed."
