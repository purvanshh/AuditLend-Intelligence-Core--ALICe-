from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.rule_sets import ACTIVE_RULE_SET, RuleSet
from ml.data.features import build_feature_row
from ml.explain.shap_explainer import explain_feature_row
from ml.governance.model_registry import ModelRegistry
from ml.models.evaluate import find_latest_manifest, load_manifest
from ml.models.train import OFFICIAL_MANIFEST_PATH
from services import FailureType, ServiceResult


DEFAULT_CREDIT_SCORE = 600
DEFAULT_INCOME_STABILITY = 0.5
ML_FEATURE_DATE = date(2026, 1, 1)
OFFICIAL_MODEL_VERSION = "XGB_V1"
_PRELOADED_ML_SCORER: "MLScorer | None" = None
_PRELOAD_ATTEMPTED = False


def compute_risk_score(
    credit_score: int | None,
    income_stability: float | None,
    dti: float,
    gst_compliant: bool | None,
    failure_types: list[FailureType],
    rule_set: RuleSet = ACTIVE_RULE_SET,
) -> tuple[float, list[str]]:
    """
    Returns (risk_score, factor_breakdown).
    risk_score is 0-100 where higher = better.
    """
    effective_credit_score = DEFAULT_CREDIT_SCORE if credit_score is None else credit_score
    effective_stability = DEFAULT_INCOME_STABILITY if income_stability is None else income_stability

    credit_component = _clamp(effective_credit_score / 900, 0.0, 1.0) * rule_set.credit_weight
    stability_component = _clamp(effective_stability, 0.0, 1.0) * rule_set.stability_weight
    dti_component = max(0.0, 1 - dti) * rule_set.dti_weight
    gst_component = rule_set.gst_weight if gst_compliant is True else 0.0
    penalty = min(
        len(failure_types) * rule_set.data_quality_penalty,
        rule_set.max_data_quality_penalty,
    )

    score = credit_component + stability_component + dti_component + gst_component - penalty
    risk_score = round(_clamp(score, 0.0, 100.0), 2)

    breakdown = [
        f"risk_score (computed) = {risk_score:.2f}",
        (
            f"credit_component ({_source_label(credit_score, 'fallback', 'live')}) = "
            f"{credit_component:.2f}/{rule_set.credit_weight:.2f} (credit_score={effective_credit_score})"
        ),
        (
            f"income_stability_component ({_source_label(income_stability, 'default', 'live')}) = "
            f"{stability_component:.2f}/{rule_set.stability_weight:.2f} "
            f"(income_stability={effective_stability:.2f})"
        ),
        f"dti_component (computed) = {dti_component:.2f}/{rule_set.dti_weight:.2f} (dti={dti:.2f})",
        f"gst_component ({_gst_label(gst_compliant)}) = {gst_component:.2f}/{rule_set.gst_weight:.2f}",
        f"data_quality_penalty (computed) = -{penalty:.2f}",
    ]
    return risk_score, breakdown


@dataclass(frozen=True)
class MLScoringResult:
    attempted: bool
    used: bool
    fallback_used: bool
    fallback_reason: str | None
    error_type: str | None
    risk_score: float | None
    predicted_default_probability: float | None
    calibrated_default_probability: float | None
    model_confidence: float | None
    model_version: str | None
    selected_candidate: str | None
    score_breakdown: list[str]
    model_factor_contributions: list[dict[str, Any]]
    model_summary: str | None

    def to_audit_output(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_factor_contributions"] = [dict(row) for row in self.model_factor_contributions]
        return payload


class MLScorer:
    """Artifact-backed ML scorer with deterministic feature mapping and guardrails."""

    def __init__(self, manifest_path: str | Path, *, model_version: str | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.uses_official_manifest = "model_artifact_path" in self.manifest
        self.model_version = model_version or str(
            self.manifest.get("model_version") or self.manifest.get("run_id") or OFFICIAL_MODEL_VERSION
        )
        self.selected_candidate = str(
            self.manifest.get("selected_candidate") or self.manifest.get("model_version") or "xgboost"
        )

    def score(
        self,
        credit_result: ServiceResult,
        bank_result: ServiceResult,
        gst_result: ServiceResult,
        user_data: dict[str, Any],
        *,
        confidence_threshold: float,
        failure_mode: str | None = None,
    ) -> MLScoringResult:
        """Score an application with the calibrated ML model or return a deterministic fallback."""

        normalized_failure_mode = str(failure_mode or "").strip().upper() or None
        if normalized_failure_mode == "TIMEOUT":
            return MLScoringResult(
                attempted=True,
                used=False,
                fallback_used=True,
                fallback_reason="TIMEOUT",
                error_type="TIMEOUT",
                risk_score=None,
                predicted_default_probability=None,
                calibrated_default_probability=None,
                model_confidence=None,
                model_version=self.model_version,
                selected_candidate=self.selected_candidate,
                score_breakdown=["ml_guardrail_fallback (applied) = TIMEOUT"],
                model_factor_contributions=[],
                model_summary="ML scoring timed out, so the heuristic scorer was used instead.",
            )

        if normalized_failure_mode in {"FORCE_CONFIDENCE_0.4", "FORCE_LOW_CONFIDENCE"}:
            return MLScoringResult(
                attempted=True,
                used=False,
                fallback_used=True,
                fallback_reason="FORCE_LOW_CONFIDENCE",
                error_type=None,
                risk_score=None,
                predicted_default_probability=0.4,
                calibrated_default_probability=0.4,
                model_confidence=0.4,
                model_version=self.model_version,
                selected_candidate=self.selected_candidate,
                score_breakdown=[
                    "ml_default_probability (forced) = 0.4000",
                    "ml_confidence (forced) = 0.4000",
                    f"ml_guardrail_fallback (applied) = model_confidence_below_threshold<{confidence_threshold:.2f}",
                ],
                model_factor_contributions=[],
                model_summary="ML confidence was forced low for testing, so the heuristic scorer was used instead.",
            )

        feature_row = _build_ml_feature_row(credit_result, bank_result, gst_result, user_data)
        explanation = explain_feature_row(feature_row, self.manifest_path, max_features=5)
        calibrated_probability = (
            explanation.calibrated_default_probability
            if explanation.calibrated_default_probability is not None
            else explanation.predicted_default_probability
        )
        model_confidence = round(max(calibrated_probability, 1.0 - calibrated_probability), 6)
        risk_score = round((1.0 - calibrated_probability) * 100.0, 2)
        fallback_used = model_confidence < confidence_threshold
        fallback_reason = (
            f"model_confidence_below_threshold<{confidence_threshold:.2f}"
            if fallback_used
            else None
        )

        score_breakdown = [
            f"ml_default_probability (raw) = {explanation.predicted_default_probability:.4f}",
            f"ml_default_probability (calibrated) = {calibrated_probability:.4f}",
            f"ml_confidence (derived) = {model_confidence:.4f}",
            f"risk_score (ml_mapped) = {risk_score:.2f}",
            f"model_version (ml) = {self.model_version}",
            f"model_candidate (ml) = {self.selected_candidate}",
        ]
        if fallback_reason is not None:
            score_breakdown.append(f"ml_guardrail_fallback (applied) = {fallback_reason}")

        return MLScoringResult(
            attempted=True,
            used=not fallback_used,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            error_type=None,
            risk_score=None if fallback_used else risk_score,
            predicted_default_probability=explanation.predicted_default_probability,
            calibrated_default_probability=calibrated_probability,
            model_confidence=model_confidence,
            model_version=self.model_version,
            selected_candidate=self.selected_candidate,
            score_breakdown=score_breakdown,
            model_factor_contributions=[dict(row) for row in explanation.to_audit_payload()["model_factor_contributions"]],
            model_summary=explanation.model_summary,
        )


def ml_scoring_requested_from_env() -> bool:
    return _env_truthy(os.getenv("ML_ENABLED")) or os.getenv("RULE_SET_VERSION", "RULE_SET_V1") == "RULE_SET_V2"


def clear_ml_scorer_cache() -> None:
    global _PRELOADED_ML_SCORER, _PRELOAD_ATTEMPTED
    get_ml_scorer_from_env.cache_clear()
    _PRELOADED_ML_SCORER = None
    _PRELOAD_ATTEMPTED = False


def preload_ml_scorer_from_env() -> MLScorer | None:
    """Resolve and freeze the ML scorer at worker startup when ML is enabled."""

    global _PRELOADED_ML_SCORER, _PRELOAD_ATTEMPTED
    ml_requested = ml_scoring_requested_from_env() or bool(os.getenv("ML_MODEL_MANIFEST_PATH") or os.getenv("ML_MODEL_VERSION"))
    if not ml_requested:
        return None
    _PRELOAD_ATTEMPTED = True
    _PRELOADED_ML_SCORER = _load_ml_scorer_from_env(allow_latest_manifest_fallback=False)
    return _PRELOADED_ML_SCORER


@lru_cache(maxsize=1)
def get_ml_scorer_from_env() -> MLScorer | None:
    if _PRELOAD_ATTEMPTED:
        return _PRELOADED_ML_SCORER
    return _load_ml_scorer_from_env()


def _load_ml_scorer_from_env(*, allow_latest_manifest_fallback: bool = True) -> MLScorer | None:
    model_version = os.getenv("ML_MODEL_VERSION")
    manifest_override = os.getenv("ML_MODEL_MANIFEST_PATH")
    if manifest_override:
        manifest_path = Path(manifest_override)
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
            resolved_version = model_version or str(manifest.get("model_version") or manifest.get("run_id") or OFFICIAL_MODEL_VERSION)
            return MLScorer(manifest_path, model_version=resolved_version)
        return None

    if OFFICIAL_MANIFEST_PATH.exists():
        manifest = load_manifest(OFFICIAL_MANIFEST_PATH)
        official_artifact_path = Path(str(manifest.get("model_artifact_path", "")))
        if official_artifact_path.exists():
            resolved_version = model_version or str(manifest.get("model_version") or OFFICIAL_MODEL_VERSION)
            return MLScorer(OFFICIAL_MANIFEST_PATH, model_version=resolved_version)
        if ml_scoring_requested_from_env():
            return None

    registry_path = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/governance/model_registry.json"))
    if registry_path.exists():
        registry = ModelRegistry(registry_path)
        try:
            record = registry.get(model_version) if model_version else registry.latest()
        except KeyError:
            record = None
        if record is not None:
            return MLScorer(record.manifest_path, model_version=record.model_version)

    if not allow_latest_manifest_fallback:
        return None

    try:
        manifest_path = find_latest_manifest()
    except FileNotFoundError:
        return None

    manifest = load_manifest(manifest_path)
    return MLScorer(manifest_path, model_version=model_version or str(manifest.get("run_id") or OFFICIAL_MODEL_VERSION))


def _build_ml_feature_row(
    credit_result: ServiceResult,
    bank_result: ServiceResult,
    gst_result: ServiceResult,
    user_data: dict[str, Any],
) -> dict[str, Any]:
    # AuditLend live applications do not carry the full Lending Club training schema.
    # We map the available deterministic fields into the model's expected feature
    # surface and use conservative proxies for fields that do not exist at decision
    # time, such as revolving-trade history and account counts.
    monthly_income = float(user_data["monthly_income"])
    existing_emis = float(user_data.get("existing_emis", 0.0))
    loan_amount = float(user_data.get("loan_amount", 0.0))
    tenure_months = max(int(user_data.get("tenure_months", 12) or 12), 1)
    installment = round(loan_amount / tenure_months, 2)

    credit_score = _extract_credit_score_value(credit_result)
    bank_data = bank_result.data or {}
    gst_data = gst_result.data or {}
    # When bank-derived subfields are missing, we fall back to declared user inputs
    # so inference remains deterministic and auditable rather than silently sparse.
    monthly_inflow = float(bank_data.get("monthly_inflow") or monthly_income)
    monthly_outflow = float(bank_data.get("monthly_outflow") or existing_emis)
    average_balance = float(bank_data.get("average_balance") or 0.0)
    income_stability = float(bank_data.get("income_stability") or DEFAULT_INCOME_STABILITY)
    purpose = str(user_data.get("purpose") or "debt_consolidation")
    home_ownership = str(user_data.get("home_ownership") or "UNKNOWN")
    verification_status = _verification_status(bank_data)
    grade, sub_grade = _grade_from_credit_score(credit_score)
    revolving_util_pct = round((1.0 - income_stability) * 100.0, 2)
    gst_compliant = bool(gst_data.get("gst_compliant")) if "gst_compliant" in gst_data else None

    clean_row = {
        "loan_id": str(user_data.get("pan_hash") or user_data.get("pan") or "auditlend-inference"),
        "issue_date": ML_FEATURE_DATE,
        "loan_status": "Current",
        "grade": grade,
        "sub_grade": sub_grade,
        "purpose": purpose,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "loan_amount": loan_amount,
        "funded_amount": loan_amount,
        "term_months": tenure_months,
        "interest_rate_pct": 0.0,
        "installment": installment,
        "monthly_income": monthly_income,
        "estimated_existing_emi": existing_emis,
        "dti_pct": _safe_ratio(existing_emis, monthly_income) * 100.0,
        "fico_midpoint": float(credit_score),
        "last_fico_midpoint": float(credit_score),
        "employment_length_years": 0.0,
        "earliest_credit_line": ML_FEATURE_DATE,
        "revol_util_pct": revolving_util_pct,
        "bc_util_pct": revolving_util_pct,
        "all_util_pct": _safe_ratio(monthly_outflow, monthly_inflow) * 100.0,
        "il_util_pct": _safe_ratio(existing_emis, monthly_income) * 100.0,
        "revol_bal": max(monthly_outflow, 0.0),
        "tot_cur_bal": average_balance,
        "total_bal_ex_mort": max(monthly_outflow, 0.0),
        "total_rev_hi_lim": max(loan_amount, 1.0),
        "total_bc_limit": max(loan_amount * 0.5, 1.0),
        "delinq_2yrs": 0.0,
        "inq_last_6mths": 0.0,
        "inq_last_12m": 0.0,
        "open_acc": 1.0,
        "total_acc": 1.0,
        "mort_acc": 1.0 if home_ownership == "MORTGAGE" else 0.0,
        "pub_rec_bankruptcies": 0.0,
        "tax_liens": 0.0 if gst_compliant is not False else 1.0,
        "percent_bc_gt_75": revolving_util_pct,
        "pct_tl_nvr_dlq": 100.0,
        "collections_12_mths_ex_med": 0.0,
        "mo_sin_rcnt_rev_tl_op": 0.0,
        "mo_sin_old_rev_tl_op": 0.0,
        "open_rv_24m": 0.0,
        "open_il_24m": 0.0,
        "defaulted": 0,
    }
    return build_feature_row(clean_row)


def _source_label(value: object | None, missing_label: str, present_label: str) -> str:
    return missing_label if value is None else present_label


def _gst_label(gst_compliant: bool | None) -> str:
    if gst_compliant is True:
        return "compliant"
    if gst_compliant is False:
        return "non_compliant"
    return "unknown"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_credit_score_value(result: ServiceResult) -> int:
    if result.data and "credit_score" in result.data:
        return int(result.data["credit_score"])
    return DEFAULT_CREDIT_SCORE


def _verification_status(bank_data: dict[str, Any]) -> str:
    if "income_stability" in bank_data:
        return "Verified"
    if "monthly_inflow" in bank_data:
        return "Source Verified"
    return "Not Verified"


def _grade_from_credit_score(credit_score: int) -> tuple[str, str]:
    if credit_score >= 780:
        return "A", "A1"
    if credit_score >= 720:
        return "B", "B2"
    if credit_score >= 660:
        return "C", "C3"
    if credit_score >= 600:
        return "D", "D4"
    return "E", "E5"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
