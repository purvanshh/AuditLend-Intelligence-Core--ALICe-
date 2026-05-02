import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ml.governance.ab_test import ExperimentAssignment, assignment_from_env
from engine.confidence import compute_data_reliability, compute_decision_confidence
from engine.rule_sets import ACTIVE_RULE_SET, RULE_SET_V2, RuleSet
from engine.rules import Decision, evaluate
from engine.scoring import MLScorer, MLScoringResult, compute_risk_score, get_ml_scorer_from_env, ml_scoring_requested_from_env
from services import FailureType, ServiceResult


@dataclass(frozen=True)
class DecisionOutput:
    decision: Decision
    confidence: float
    data_reliability: float
    risk_score: float
    factors: list[str]
    penalty_reasons: list[str]
    rule_version: str
    requires_manual_review: bool
    scoring_strategy: str = "heuristic"
    model_version: str | None = None
    selected_candidate: str | None = None
    ml_requested: bool = False
    ml_fallback_used: bool = False
    ml_fallback_reason: str | None = None
    ml_error_type: str | None = None
    ml_default_probability: float | None = None
    ml_confidence: float | None = None
    model_summary: str | None = None
    model_factor_contributions: list[dict[str, Any]] = field(default_factory=list)
    ml_drift_report: dict[str, Any] | None = None
    ab_test_arm: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload


def compute_decision(
    credit_result: ServiceResult,
    bank_result: ServiceResult,
    gst_result: ServiceResult,
    user_data: dict[str, Any],
    confidence_threshold: float = 0.6,
    rule_set: RuleSet = ACTIVE_RULE_SET,
    *,
    ml_scorer: MLScorer | None = None,
    ml_requested: bool = False,
    ml_failure_mode: str | None = None,
    ab_test_assignment: ExperimentAssignment | None = None,
) -> DecisionOutput:
    """
    Orchestrates extraction, risk scoring, calibrated confidence, and manual review override.
    This function is deterministic and side-effect free.
    """
    monthly_income = float(user_data["monthly_income"])
    existing_emis = float(user_data.get("existing_emis", 0))
    dti = existing_emis / monthly_income

    credit_score = _extract_credit_score(credit_result)
    income_stability = _extract_income_stability(bank_result)
    gst_compliant = _extract_gst_compliance(gst_result)

    failure_types = _collect_failure_types(credit_result, bank_result, gst_result)
    used_fallback_credit = (
        credit_result.fallback_used
        and credit_result.failure_type in {FailureType.TIMEOUT, FailureType.SERVICE_DOWN}
    )

    heuristic_risk_score, heuristic_score_breakdown = compute_risk_score(
        credit_score,
        income_stability,
        dti,
        gst_compliant,
        failure_types,
        rule_set,
    )
    ml_result = _maybe_score_with_ml(
        ml_requested=ml_requested,
        ml_scorer=ml_scorer,
        ml_failure_mode=ml_failure_mode,
        credit_result=credit_result,
        bank_result=bank_result,
        gst_result=gst_result,
        user_data=user_data,
        confidence_threshold=confidence_threshold,
    )
    rule_set_for_decision = RULE_SET_V2 if ml_result and ml_result.used else rule_set
    rule_version = rule_set_for_decision.version if ml_result and ml_result.used else rule_set.version
    scoring_strategy = "ml" if ml_result and ml_result.used else "heuristic"
    risk_score = ml_result.risk_score if ml_result and ml_result.used and ml_result.risk_score is not None else heuristic_risk_score
    score_breakdown = list(heuristic_score_breakdown)
    if ml_result and ml_result.used:
        score_breakdown = [f"heuristic_risk_score (shadow) = {heuristic_risk_score:.2f}"] + ml_result.score_breakdown
    elif ml_result and ml_result.attempted:
        score_breakdown.extend(ml_result.score_breakdown)

    decision, factors = evaluate(
        risk_score,
        credit_score,
        dti,
        failure_types,
        gst_compliant,
        rule_set_for_decision,
    )
    factors = score_breakdown + factors

    data_reliability, penalty_reasons = compute_data_reliability(failure_types, used_fallback_credit)
    confidence, confidence_reasons = compute_decision_confidence(
        risk_score,
        decision,
        data_reliability,
        failure_types,
    )
    penalty_reasons.extend(confidence_reasons)

    requires_manual_review = confidence < confidence_threshold
    if requires_manual_review:
        decision = Decision.NEEDS_REVIEW
        factors.append("Confidence below threshold - routed to manual review")

    return DecisionOutput(
        decision=decision,
        confidence=confidence,
        data_reliability=data_reliability,
        risk_score=risk_score,
        factors=factors,
        penalty_reasons=penalty_reasons,
        rule_version=rule_version,
        requires_manual_review=requires_manual_review,
        scoring_strategy=scoring_strategy,
        model_version=ml_result.model_version if ml_result else None,
        selected_candidate=ml_result.selected_candidate if ml_result else None,
        ml_requested=ml_requested,
        ml_fallback_used=ml_result.fallback_used if ml_result else False,
        ml_fallback_reason=ml_result.fallback_reason if ml_result else None,
        ml_error_type=ml_result.error_type if ml_result else None,
        ml_default_probability=ml_result.calibrated_default_probability if ml_result else None,
        ml_confidence=ml_result.model_confidence if ml_result else None,
        model_summary=ml_result.model_summary if ml_result else None,
        model_factor_contributions=ml_result.model_factor_contributions if ml_result else [],
        ml_drift_report=ml_result.drift_report if ml_result else None,
        ab_test_arm=ab_test_assignment.arm if ab_test_assignment is not None else None,
    )


def compute_decision_from_env(
    credit_result: ServiceResult,
    bank_result: ServiceResult,
    gst_result: ServiceResult,
    user_data: dict[str, Any],
    failure_flags: dict[str, Any] | None = None,
    application_id: str | None = None,
) -> DecisionOutput:
    assignment = assignment_from_env(application_id) if application_id is not None else None
    ml_requested = _ml_requested_from_env_and_assignment(assignment)
    return compute_decision(
        credit_result,
        bank_result,
        gst_result,
        user_data,
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
        rule_set=ACTIVE_RULE_SET,
        ml_scorer=get_ml_scorer_from_env() if ml_requested else None,
        ml_requested=ml_requested,
        ml_failure_mode=(failure_flags or {}).get("ml_model"),
        ab_test_assignment=assignment,
    )


def _collect_failure_types(*results: ServiceResult) -> list[FailureType]:
    return [result.failure_type for result in results if result.failure_type is not None]


def _extract_credit_score(result: ServiceResult) -> int | None:
    if result.fallback_used and result.failure_type in {FailureType.TIMEOUT, FailureType.SERVICE_DOWN}:
        return None
    if result.data and "credit_score" in result.data:
        return int(result.data["credit_score"])
    return None


def _extract_income_stability(result: ServiceResult) -> float | None:
    if result.data and "income_stability" in result.data:
        return float(result.data["income_stability"])
    return None


def _extract_gst_compliance(result: ServiceResult) -> bool | None:
    if result.data and "gst_compliant" in result.data:
        return bool(result.data["gst_compliant"])
    return False if result.failure_type in {FailureType.PAN_MISMATCH, FailureType.NO_RECORD} else None


def _maybe_score_with_ml(
    *,
    ml_requested: bool,
    ml_scorer: MLScorer | None,
    ml_failure_mode: str | None,
    credit_result: ServiceResult,
    bank_result: ServiceResult,
    gst_result: ServiceResult,
    user_data: dict[str, Any],
    confidence_threshold: float,
) -> MLScoringResult | None:
    if not ml_requested:
        return None
    if ml_scorer is None:
        return MLScoringResult(
            attempted=True,
            used=False,
            fallback_used=True,
            fallback_reason="MODEL_UNAVAILABLE",
            error_type="MODEL_UNAVAILABLE",
            risk_score=None,
            predicted_default_probability=None,
            calibrated_default_probability=None,
            model_confidence=None,
            model_version=None,
            selected_candidate=None,
            score_breakdown=["ml_guardrail_fallback (applied) = MODEL_UNAVAILABLE"],
            model_factor_contributions=[],
            model_summary="ML model artifacts were unavailable, so the heuristic scorer was used instead.",
            drift_report=None,
        )
    return ml_scorer.score(
        credit_result,
        bank_result,
        gst_result,
        user_data,
        confidence_threshold=confidence_threshold,
        failure_mode=ml_failure_mode,
    )


def _ml_requested_from_env_and_assignment(assignment: ExperimentAssignment | None) -> bool:
    explicit_request = ml_scoring_requested_from_env()
    if assignment is not None:
        return assignment.arm == "ml"
    return explicit_request
