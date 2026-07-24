"""Portfolio Analysis Demo – deterministic synthetic data, no external dependencies."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from typing import Any

from ml.portfolio.risk_aggregator import (
    PortfolioSummary,
    compute_portfolio_summary,
    concentration_analysis,
    run_stress_test,
)
from ml.portfolio.report_generator import (
    generate_portfolio_report,
    portfolio_report_to_html,
)


def _deterministic_decision(seed_bytes: bytes, index: int) -> dict[str, Any]:
    h = sha256(seed_bytes + str(index).encode()).hexdigest()
    risk_raw = int(h[:8], 16)
    risk_score = (risk_raw % 10000) / 100.0

    amount_raw = int(h[8:16], 16)
    loan_amount = 5000 + (amount_raw % 450000)

    decision_seed = int(h[16:24], 16) % 100
    if risk_score < 30 and decision_seed < 90:
        decision = "APPROVED"
    elif risk_score > 70 and decision_seed < 80:
        decision = "DECLINED"
    else:
        decision = "NEEDS_REVIEW"

    industry_seed = int(h[24:32], 16) % 10
    industries = ["technology", "healthcare", "finance", "manufacturing", "retail",
                  "energy", "real_estate", "transportation", "agriculture", "education"]
    industry = industries[industry_seed]

    return {
        "risk_score": risk_score,
        "loan_amount": loan_amount,
        "decision": decision,
        "confidence": 0.75 + (risk_raw % 2000) / 10000.0,
        "industry": industry,
    }


def generate_synthetic_portfolio(n: int = 500, seed: str = "portfolio-demo-2026") -> list[dict[str, Any]]:
    seed_bytes = seed.encode()
    return [_deterministic_decision(seed_bytes, i) for i in range(n)]


def demo() -> None:
    print("=" * 68)
    print("  AuditLend Portfolio Analysis Demo")
    print("=" * 68)

    decisions = generate_synthetic_portfolio(500)
    print(f"\nGenerated {len(decisions)} synthetic decisions (deterministic)\n")

    print("-" * 68)
    print("  PORTFOLIO SUMMARY")
    print("-" * 68)
    summary = compute_portfolio_summary(decisions)
    print(f"  Total Applications:      {summary.total_applications}")
    print(f"  Approved:                {summary.approved_count}")
    print(f"  Declined:                {summary.declined_count}")
    print(f"  Manual Review:           {summary.manual_review_count}")
    print(f"  Approval Rate:           {summary.approval_rate:.2%}")
    print(f"  Decline Rate:            {summary.decline_rate:.2%}")
    print(f"  Manual Review Rate:      {summary.manual_review_rate:.2%}")
    print(f"  Total Loan Amount:       ${summary.total_loan_amount:,.2f}")
    print(f"  Avg Risk Score:          {summary.avg_risk_score:.2f}")
    print(f"  Median Risk Score:       {summary.median_risk_score:.2f}")
    print(f"  Std Dev Risk Score:      {summary.std_risk_score:.2f}")
    print(f"  Percentiles:             {summary.percentile_risk_scores}")
    print(f"  Risk Buckets:            {summary.risk_buckets}")

    print(f"\n{'=' * 68}")
    print("  STRESS TESTING")
    print("=" * 68)
    scenarios = [
        ("Mild Recession", 0.10),
        ("Severe Recession", 0.25),
        ("Interest Rate Shock", 0.15),
    ]
    for name, shock in scenarios:
        result = run_stress_test(decisions, scenario_name=name, shock_factor=shock)
        print(f"\n  Scenario: {name} (+{shock:.0%} shock)")
        print(f"    Baseline Approval:  {result.baseline_approval_rate:.2%}")
        print(f"    Stressed Approval:  {result.stressed_approval_rate:.2%}")
        print(f"    Baseline Loss Rate: {result.baseline_loss_rate:.2%}")
        print(f"    Stressed Loss Rate: {result.stressed_loss_rate:.2%}")
        print(f"    Additional Loss:    {result.additional_loss_pct:+.2f}%")

    print(f"\n{'=' * 68}")
    print("  CONCENTRATION ANALYSIS")
    print("=" * 68)
    conc = concentration_analysis(decisions, segment_key="industry")
    print(f"  Segment Key: {conc['segment']}")
    print(f"  HHI:          {conc['hhi']}")
    print(f"  Interpretation: {conc['interpretation']}")
    print(f"  Top Segments:")
    for seg in conc["top_segments"]:
        print(f"    - {seg['segment']}: {seg['exposure_pct']:.2f}%")

    print(f"\n{'=' * 68}")
    print("  REPORT GENERATION")
    print("=" * 68)
    md_path = generate_portfolio_report(summary, "/tmp/portfolio_report_demo.md")
    print(f"  Markdown report: {md_path}")
    html_path = portfolio_report_to_html(summary, "/tmp/portfolio_report_demo.html")
    print(f"  HTML report:     {html_path}")
    print(f"\n{'=' * 68}")
    print("  Demo complete.")
    print("=" * 68)


if __name__ == "__main__":
    demo()
