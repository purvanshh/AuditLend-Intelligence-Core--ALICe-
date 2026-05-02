from engine.rules import Decision
from engine.scoring import MLScoringResult
from engine.decision import compute_decision, compute_decision_from_env
from services import FailureType, ServiceResult


USER_DATA = {
    "monthly_income": 120000,
    "existing_emis": 25000,
    "loan_amount": 500000,
    "tenure_months": 36,
}


def result(data: dict | None, failure_type: FailureType | None = None, fallback_used: bool = False) -> ServiceResult:
    return ServiceResult(
        success=failure_type is None,
        data=data,
        failure_type=failure_type,
        raw_response=data,
        fallback_used=fallback_used,
    )


def test_high_quality_profile_is_approved_with_full_confidence() -> None:
    output = compute_decision(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.decision == Decision.APPROVE
    assert output.confidence == 1.0
    assert output.data_reliability == 1.0
    assert output.risk_score == 88.35
    assert output.requires_manual_review is False


def test_credit_timeout_uses_fallback_score_and_manual_review_override() -> None:
    output = compute_decision(
        result({"credit_score": 600}, FailureType.TIMEOUT, fallback_used=True),
        result({"income_stability": 0.8}),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.decision == Decision.NEEDS_REVIEW
    assert output.confidence == 0.54
    assert output.data_reliability == 0.6
    assert output.risk_score == 72.46
    assert output.requires_manual_review is True
    assert "fallback_credit_score: -0.10" in output.penalty_reasons
    assert "credit_component (fallback) = 26.67/40.00 (credit_score=600)" in output.factors


def test_low_confidence_overrides_high_rule_score_to_manual_review() -> None:
    output = compute_decision(
        result({"credit_score": 850}, FailureType.STALE_DATA),
        result({"monthly_inflow": 120000}, FailureType.PARTIAL_DATA),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.decision == Decision.NEEDS_REVIEW
    assert output.data_reliability == 0.6
    assert output.confidence == 0.54
    assert output.requires_manual_review is True
    assert output.factors[-1] == "Confidence below threshold - routed to manual review"


def test_confidence_exactly_at_threshold_is_not_overridden() -> None:
    output = compute_decision(
        result({"credit_score": 900}, FailureType.STALE_DATA),
        result({"income_stability": 1.0}, FailureType.PARTIAL_DATA),
        result({"gst_compliant": True}),
        {"monthly_income": 120000, "existing_emis": 0, "loan_amount": 500000, "tenure_months": 36},
        confidence_threshold=0.6,
    )

    assert output.confidence == 0.6
    assert output.requires_manual_review is False


def test_high_dti_declines_when_confidence_is_high_enough() -> None:
    output = compute_decision(
        result({"credit_score": 300}),
        result({"income_stability": 0.0}),
        result({"gst_compliant": True}),
        {"monthly_income": 120000, "existing_emis": 84000, "loan_amount": 500000, "tenure_months": 36},
    )

    assert output.decision == Decision.DECLINE
    assert output.confidence == 0.75


def test_partial_bank_data_uses_neutral_stability() -> None:
    output = compute_decision(
        result({"credit_score": 700}),
        result({"monthly_inflow": 120000}, FailureType.PARTIAL_DATA),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.decision == Decision.APPROVE
    assert "income_stability_component (default) = 10.00/20.00 (income_stability=0.50)" in output.factors
    assert output.data_reliability == 0.8
    assert output.confidence == 0.72


def test_decision_output_serializes_enum_values() -> None:
    output = compute_decision(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.to_dict()["decision"] == "APPROVE"
    assert output.to_dict()["risk_score"] == 88.35
    assert output.to_dict()["data_reliability"] == 1.0


def test_env_wrapper_uses_configured_threshold_and_active_rule_set(monkeypatch) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.95")

    output = compute_decision_from_env(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.rule_version == "RULE_SET_V1"
    assert output.requires_manual_review is False


def test_missing_credit_without_fallback_uses_default_source() -> None:
    output = compute_decision(
        result(None),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert "credit_component (fallback) = 26.67/40.00 (credit_score=600)" in output.factors
    assert output.risk_score == 79.46
    assert output.confidence == 0.9


def test_format_error_bank_data_uses_default_stability() -> None:
    output = compute_decision(
        result({"credit_score": 700}),
        result(None, FailureType.FORMAT_ERROR, fallback_used=True),
        result({"gst_compliant": True}),
        USER_DATA,
    )

    assert output.decision == Decision.APPROVE
    assert "income_stability_component (default) = 10.00/20.00 (income_stability=0.50)" in output.factors
    assert output.data_reliability == 0.7
    assert output.confidence == 0.63


def test_gst_no_record_defaults_to_non_compliant() -> None:
    output = compute_decision(
        result({"credit_score": 700}),
        result({"income_stability": 0.7}),
        result(None, FailureType.NO_RECORD, fallback_used=True),
        USER_DATA,
    )

    assert output.decision == Decision.NEEDS_REVIEW
    assert "gst_component (non_compliant) = 0.00/15.00" in output.factors
    assert "gst_gate (applied) = risk_score capped at 54.00" in output.factors


class FakeMLScorer:
    def __init__(self, result: MLScoringResult):
        self.result = result

    def score(self, *args, **kwargs) -> MLScoringResult:
        return self.result


def test_confident_ml_score_uses_rule_set_v2() -> None:
    ml_result = MLScoringResult(
        attempted=True,
        used=True,
        fallback_used=False,
        fallback_reason=None,
        error_type=None,
        risk_score=82.4,
        predicted_default_probability=0.19,
        calibrated_default_probability=0.176,
        model_confidence=0.824,
        model_version="XGB_V1",
        selected_candidate="lightgbm",
        score_breakdown=[
            "ml_default_probability (raw) = 0.1900",
            "ml_default_probability (calibrated) = 0.1760",
            "risk_score (ml_mapped) = 82.40",
        ],
        model_factor_contributions=[
            {
                "feature_name": "Debt-To-Income Ratio",
                "raw_value": "20.8%",
                "shap_contribution": -0.11,
                "direction": "decrease_default_risk",
            }
        ],
        model_summary="Model factors: Debt-To-Income Ratio (20.8%) reduced predicted default risk.",
    )

    output = compute_decision(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
        ml_scorer=FakeMLScorer(ml_result),
        ml_requested=True,
    )

    assert output.decision == Decision.APPROVE
    assert output.rule_version == "RULE_SET_V2"
    assert output.scoring_strategy == "ml"
    assert output.risk_score == 82.4
    assert output.model_version == "XGB_V1"
    assert output.model_factor_contributions[0]["feature_name"] == "Debt-To-Income Ratio"
    assert output.factors[0] == "heuristic_risk_score (shadow) = 88.35"


def test_low_confidence_ml_result_falls_back_to_heuristic() -> None:
    ml_result = MLScoringResult(
        attempted=True,
        used=False,
        fallback_used=True,
        fallback_reason="model_confidence_below_threshold<0.60",
        error_type=None,
        risk_score=None,
        predicted_default_probability=0.49,
        calibrated_default_probability=0.48,
        model_confidence=0.52,
        model_version="XGB_V1",
        selected_candidate="lightgbm",
        score_breakdown=[
            "ml_default_probability (raw) = 0.4900",
            "ml_default_probability (calibrated) = 0.4800",
            "ml_guardrail_fallback (applied) = model_confidence_below_threshold<0.60",
        ],
        model_factor_contributions=[],
        model_summary="ML confidence was too low, so the heuristic scorer was used instead.",
    )

    output = compute_decision(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
        ml_scorer=FakeMLScorer(ml_result),
        ml_requested=True,
    )

    assert output.decision == Decision.APPROVE
    assert output.rule_version == "RULE_SET_V1"
    assert output.scoring_strategy == "heuristic"
    assert output.risk_score == 88.35
    assert output.ml_fallback_used is True
    assert output.ml_fallback_reason == "model_confidence_below_threshold<0.60"
    assert "ml_guardrail_fallback (applied) = model_confidence_below_threshold<0.60" in output.factors


def test_env_wrapper_respects_ab_test_assignment(monkeypatch) -> None:
    ml_result = MLScoringResult(
        attempted=True,
        used=True,
        fallback_used=False,
        fallback_reason=None,
        error_type=None,
        risk_score=80.0,
        predicted_default_probability=0.2,
        calibrated_default_probability=0.2,
        model_confidence=0.8,
        model_version="XGB_V1",
        selected_candidate="lightgbm",
        score_breakdown=["risk_score (ml_mapped) = 80.00"],
        model_factor_contributions=[],
        model_summary="Model factors: Credit Score (800) reduced predicted default risk.",
    )

    monkeypatch.setenv("AB_TEST_ENABLED", "true")
    monkeypatch.setenv("AB_TEST_ML_RATIO", "1.0")
    monkeypatch.setattr("engine.decision.get_ml_scorer_from_env", lambda: FakeMLScorer(ml_result))

    output = compute_decision_from_env(
        result({"credit_score": 800}),
        result({"income_stability": 0.9}),
        result({"gst_compliant": True}),
        USER_DATA,
        application_id="ab-test-app-1",
    )

    assert output.ab_test_arm == "ml"
    assert output.scoring_strategy == "ml"
    assert output.rule_version == "RULE_SET_V2"
