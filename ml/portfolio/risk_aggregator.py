from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median, stdev
from typing import Any


@dataclass
class PortfolioSummary:
    total_applications: int = 0
    approved_count: int = 0
    declined_count: int = 0
    manual_review_count: int = 0
    total_loan_amount: float = 0.0
    avg_risk_score: float = 0.0
    median_risk_score: float = 0.0
    std_risk_score: float = 0.0
    percentile_risk_scores: dict[str, float] = field(default_factory=dict)
    approval_rate: float = 0.0
    decline_rate: float = 0.0
    manual_review_rate: float = 0.0
    risk_buckets: dict[str, int] = field(default_factory=lambda: {
        "low": 0, "medium": 0, "high": 0, "very_high": 0,
    })


@dataclass
class PortfolioStressTestResult:
    scenario_name: str
    shock_factor: float
    baseline_approval_rate: float
    stressed_approval_rate: float
    baseline_loss_rate: float
    stressed_loss_rate: float
    additional_loss_pct: float


def _compute_percentiles(sorted_scores: list[float], percentiles: list[float]) -> dict[str, float]:
    n = len(sorted_scores)
    if n == 0:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    result: dict[str, float] = {}
    for p in percentiles:
        idx = p / 100.0 * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            result[f"p{int(p)}"] = sorted_scores[lo]
        else:
            frac = idx - lo
            result[f"p{int(p)}"] = sorted_scores[lo] * (1 - frac) + sorted_scores[hi] * frac
    return result


def compute_portfolio_summary(decisions: list[dict[str, Any]]) -> PortfolioSummary:
    summary = PortfolioSummary()
    summary.total_applications = len(decisions)

    if not decisions:
        return summary

    risk_scores: list[float] = []
    loan_amounts: list[float] = []
    approved_count = 0
    declined_count = 0
    manual_review_count = 0

    for d in decisions:
        risk_score = d.get("risk_score", 0.0)
        risk_scores.append(risk_score)
        loan_amount = d.get("loan_amount", 0.0)
        loan_amounts.append(loan_amount)
        decision = d.get("decision", "").upper()

        if decision == "APPROVED":
            approved_count += 1
        elif decision == "DECLINED":
            declined_count += 1
        else:
            manual_review_count += 1

        if risk_score <= 25:
            summary.risk_buckets["low"] += 1
        elif risk_score <= 50:
            summary.risk_buckets["medium"] += 1
        elif risk_score <= 75:
            summary.risk_buckets["high"] += 1
        else:
            summary.risk_buckets["very_high"] += 1

    summary.approved_count = approved_count
    summary.declined_count = declined_count
    summary.manual_review_count = manual_review_count
    summary.total_loan_amount = sum(loan_amounts)
    summary.approval_rate = approved_count / len(decisions) if decisions else 0.0
    summary.decline_rate = declined_count / len(decisions) if decisions else 0.0
    summary.manual_review_rate = manual_review_count / len(decisions) if decisions else 0.0

    sorted_scores = sorted(risk_scores)
    summary.avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    summary.median_risk_score = median(risk_scores) if risk_scores else 0.0
    summary.std_risk_score = stdev(risk_scores) if len(risk_scores) > 1 else 0.0
    summary.percentile_risk_scores = _compute_percentiles(sorted_scores, [5, 25, 50, 75, 95])

    return summary


def run_stress_test(
    decisions: list[dict[str, Any]],
    scenario_name: str,
    shock_factor: float,
    default_threshold: float = 0.8,
) -> PortfolioStressTestResult:
    baseline_approved = sum(1 for d in decisions if d.get("decision", "").upper() == "APPROVED")
    baseline_total = len(decisions)
    baseline_approval_rate = baseline_approved / baseline_total if baseline_total else 0.0

    baseline_losses = sum(1 for d in decisions if d.get("decision", "").upper() == "APPROVED" and d.get("risk_score", 0.0) > 0.5)
    baseline_loss_rate = baseline_losses / baseline_approved if baseline_approved else 0.0

    stressed_approved = 0
    stressed_losses = 0
    for d in decisions:
        stressed_score = d.get("risk_score", 0.0) + shock_factor
        decision = d.get("decision", "").upper()
        if decision == "APPROVED":
            if stressed_score <= default_threshold:
                stressed_approved += 1
            else:
                stressed_losses += 1
        elif decision != "DECLINED":
            if stressed_score <= default_threshold:
                stressed_approved += 1
            else:
                stressed_losses += 1

    stressed_total = len(decisions)
    stressed_approval_rate = stressed_approved / stressed_total if stressed_total else 0.0
    stressed_loss_rate = stressed_losses / stressed_approved if stressed_approved else 0.0

    additional_loss_pct = 0.0
    if baseline_loss_rate > 0:
        additional_loss_pct = ((stressed_loss_rate - baseline_loss_rate) / baseline_loss_rate) * 100

    return PortfolioStressTestResult(
        scenario_name=scenario_name,
        shock_factor=shock_factor,
        baseline_approval_rate=baseline_approval_rate,
        stressed_approval_rate=stressed_approval_rate,
        baseline_loss_rate=baseline_loss_rate,
        stressed_loss_rate=stressed_loss_rate,
        additional_loss_pct=additional_loss_pct,
    )


def concentration_analysis(
    decisions: list[dict[str, Any]],
    segment_key: str = "industry",
) -> dict[str, Any]:
    segments: dict[str, float] = {}
    total_amount = 0.0

    for d in decisions:
        seg = d.get(segment_key, "unknown")
        amount = d.get("loan_amount", 0.0)
        segments[seg] = segments.get(seg, 0.0) + amount
        total_amount += amount

    if total_amount == 0:
        return {
            "segment": segment_key,
            "hhi": 0.0,
            "top_segments": [],
            "interpretation": "no_data",
        }

    shares = [amt / total_amount for amt in segments.values()]
    hhi = sum(s * s for s in shares) * 10000

    sorted_segments = sorted(segments.items(), key=lambda x: x[1], reverse=True)
    top_segments = [
        {"segment": seg, "exposure_pct": round(amt / total_amount * 100, 2)}
        for seg, amt in sorted_segments[:5]
    ]

    if hhi < 1000:
        interpretation = "diversified"
    elif hhi < 2500:
        interpretation = "moderately_concentrated"
    else:
        interpretation = "highly_concentrated"

    return {
        "segment": segment_key,
        "hhi": round(hhi, 2),
        "top_segments": top_segments,
        "interpretation": interpretation,
    }
