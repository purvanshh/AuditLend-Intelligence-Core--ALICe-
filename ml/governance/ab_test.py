"""A/B routing and summary utilities for ML rollout governance."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DEFAULT_ML_TRAFFIC_RATIO = 0.10
AB_BUCKET_COUNT = 10_000


@dataclass(frozen=True)
class ExperimentAssignment:
    """Deterministic experiment-arm routing for one application."""

    enabled: bool
    arm: str
    ml_ratio: float
    bucket: int


@dataclass(frozen=True)
class OutcomeRecord:
    """One decision outcome used in A/B summary reporting."""

    arm: str
    decision: str
    confidence: float
    defaulted: int
    loan_amount: float
    scoring_strategy: str


@dataclass(frozen=True)
class ArmSummary:
    """Aggregate outcome metrics for one experiment arm."""

    arm: str
    row_count: int
    approval_rate: float
    decline_rate: float
    manual_review_rate: float
    average_confidence: float
    average_loan_amount: float
    approved_count: int
    default_rate_on_approved: float
    simulated_profit: float
    simulated_profit_per_application: float


@dataclass(frozen=True)
class ABTestReport:
    """Comparable summary for both experiment arms."""

    ml_ratio: float
    arms: list[ArmSummary]
    approval_rate_delta_ml_minus_heuristic: float
    default_rate_delta_ml_minus_heuristic: float
    profit_delta_ml_minus_heuristic: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arms"] = [asdict(row) for row in self.arms]
        return payload


def assign_experiment_arm(
    application_id: str,
    *,
    ml_ratio: float = DEFAULT_ML_TRAFFIC_RATIO,
    enabled: bool = True,
) -> ExperimentAssignment:
    """Assign an application deterministically to `heuristic` or `ml`."""

    normalized_ratio = min(max(float(ml_ratio), 0.0), 1.0)
    bucket = _stable_bucket(application_id)
    ml_threshold = int(round(normalized_ratio * AB_BUCKET_COUNT))
    if not enabled:
        return ExperimentAssignment(enabled=False, arm="heuristic", ml_ratio=normalized_ratio, bucket=bucket)
    arm = "ml" if bucket < ml_threshold else "heuristic"
    return ExperimentAssignment(enabled=True, arm=arm, ml_ratio=normalized_ratio, bucket=bucket)


def assignment_from_env(application_id: str) -> ExperimentAssignment | None:
    """Resolve optional A/B routing from environment variables."""

    enabled = _env_truthy(os.getenv("AB_TEST_ENABLED"))
    if not enabled:
        return None
    ratio = float(os.getenv("AB_TEST_ML_RATIO", str(DEFAULT_ML_TRAFFIC_RATIO)))
    return assign_experiment_arm(application_id, ml_ratio=ratio, enabled=True)


def summarize_outcomes(
    records: Iterable[OutcomeRecord],
    *,
    ml_ratio: float = DEFAULT_ML_TRAFFIC_RATIO,
) -> ABTestReport:
    """Aggregate A/B outcome rows into arm-level comparisons."""

    grouped: dict[str, list[OutcomeRecord]] = {"heuristic": [], "ml": []}
    for record in records:
        grouped.setdefault(record.arm, []).append(record)

    arm_summaries = [_summarize_arm(arm, grouped.get(arm, [])) for arm in ("heuristic", "ml")]
    summary_by_arm = {row.arm: row for row in arm_summaries}
    heuristic = summary_by_arm["heuristic"]
    ml = summary_by_arm["ml"]
    return ABTestReport(
        ml_ratio=ml_ratio,
        arms=arm_summaries,
        approval_rate_delta_ml_minus_heuristic=round(ml.approval_rate - heuristic.approval_rate, 6),
        default_rate_delta_ml_minus_heuristic=round(ml.default_rate_on_approved - heuristic.default_rate_on_approved, 6),
        profit_delta_ml_minus_heuristic=round(ml.simulated_profit - heuristic.simulated_profit, 2),
    )


def _summarize_arm(arm: str, rows: list[OutcomeRecord]) -> ArmSummary:
    if not rows:
        return ArmSummary(
            arm=arm,
            row_count=0,
            approval_rate=0.0,
            decline_rate=0.0,
            manual_review_rate=0.0,
            average_confidence=0.0,
            average_loan_amount=0.0,
            approved_count=0,
            default_rate_on_approved=0.0,
            simulated_profit=0.0,
            simulated_profit_per_application=0.0,
        )

    approvals = [row for row in rows if row.decision == "APPROVE"]
    declines = [row for row in rows if row.decision == "DECLINE"]
    reviews = [row for row in rows if row.decision == "NEEDS_REVIEW"]
    approved_defaults = [row for row in approvals if int(row.defaulted) == 1]
    simulated_profit = sum(_simulated_profit(row.loan_amount, row.defaulted) for row in approvals)

    row_count = len(rows)
    average_confidence = sum(float(row.confidence) for row in rows) / row_count
    average_loan_amount = sum(float(row.loan_amount) for row in rows) / row_count
    approved_count = len(approvals)

    return ArmSummary(
        arm=arm,
        row_count=row_count,
        approval_rate=round(approved_count / row_count, 6),
        decline_rate=round(len(declines) / row_count, 6),
        manual_review_rate=round(len(reviews) / row_count, 6),
        average_confidence=round(average_confidence, 6),
        average_loan_amount=round(average_loan_amount, 2),
        approved_count=approved_count,
        default_rate_on_approved=round((len(approved_defaults) / approved_count) if approved_count else 0.0, 6),
        simulated_profit=round(simulated_profit, 2),
        simulated_profit_per_application=round(simulated_profit / row_count, 2),
    )


def _simulated_profit(loan_amount: float, defaulted: int) -> float:
    paid_profit = float(loan_amount) * 0.12
    default_loss = float(loan_amount) * 0.65
    return -default_loss if int(defaulted) == 1 else paid_profit


def _stable_bucket(application_id: str) -> int:
    digest = hashlib.sha256(application_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % AB_BUCKET_COUNT


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
