from datetime import UTC, datetime

from engine.explanation_builder import build_explanation
from models.audit_log import AuditLog


def test_build_explanation_from_degraded_audit_trail() -> None:
    entries = [
        AuditLog(
            step="CREDIT_BUREAU_FETCH",
            error_type="TIMEOUT",
            fallback_used=True,
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
        AuditLog(
            step="DECISION_CALCULATION",
            output_snapshot={
                "decision": "NEEDS_REVIEW",
                "confidence": 0.5,
                "factors": [
                    "credit_score (fallback) = 600",
                    "income_stability (live) = 0.8",
                    "dti (computed) = 0.25",
                    "Confidence below threshold - routed to manual review",
                ],
                "rule_version": "RULE_SET_V1",
            },
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
    ]

    explanation = build_explanation(entries, entries[-1].output_snapshot)

    assert explanation["decision"] == "NEEDS_REVIEW"
    assert "insufficient reliable data" in explanation["summary"]
    assert explanation["factors"][0] == {
        "name": "Credit Score",
        "value": "600",
        "status": "fallback",
    }
    assert explanation["timeline"][0]["status"] == "TIMEOUT"


def test_build_explanation_includes_ml_contributions_from_audit_trail() -> None:
    entries = [
        AuditLog(
            step="ML_SCORING",
            output_snapshot={
                "model_version": "XGB_V1",
                "model_summary": (
                    "Model factors: Debt-To-Income Ratio (42.0%) increased predicted default risk, "
                    "while Credit Score (720) reduced predicted default risk."
                ),
                "model_factor_contributions": [
                    {
                        "feature_name": "Debt-To-Income Ratio",
                        "raw_value": "42.0%",
                        "shap_contribution": 0.182,
                        "direction": "increase_default_risk",
                    },
                    {
                        "feature_name": "Credit Score",
                        "raw_value": "720",
                        "shap_contribution": -0.149,
                        "direction": "decrease_default_risk",
                    },
                ],
            },
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
        AuditLog(
            step="DECISION_CALCULATION",
            output_snapshot={
                "decision": "DECLINE",
                "confidence": 0.82,
                "factors": [
                    "risk_score (computed) = 28.00",
                    "dti (computed) = 0.42",
                ],
                "rule_version": "RULE_SET_V2",
            },
            created_at=datetime(2026, 4, 26, tzinfo=UTC),
        ),
    ]

    explanation = build_explanation(entries, entries[-1].output_snapshot)

    assert explanation["decision"] == "DECLINE"
    assert explanation["model_version"] == "XGB_V1"
    assert len(explanation["model_factor_contributions"]) == 2
    assert "Model factors:" in explanation["summary"]
    assert explanation["model_factor_contributions"][0]["feature_name"] == "Debt-To-Income Ratio"
