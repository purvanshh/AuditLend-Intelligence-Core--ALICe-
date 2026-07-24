"""Unit tests for ml/portfolio/risk_aggregator.py."""

from __future__ import annotations

import math
from typing import Any

from ml.portfolio.risk_aggregator import (
    PortfolioSummary,
    PortfolioStressTestResult,
    compute_portfolio_summary,
    concentration_analysis,
    run_stress_test,
)


def _sample_decisions(n: int = 10) -> list[dict[str, Any]]:
    return [
        {"risk_score": i * 10.0, "loan_amount": 10000.0 + i * 1000, "decision": "APPROVED", "industry": "tech"}
        for i in range(n)
    ]


class TestPortfolioSummary:

    def test_empty_decisions(self) -> None:
        summary = compute_portfolio_summary([])
        assert summary.total_applications == 0
        assert summary.approved_count == 0
        assert summary.declined_count == 0
        assert summary.manual_review_count == 0
        assert summary.total_loan_amount == 0.0
        assert summary.avg_risk_score == 0.0
        assert summary.median_risk_score == 0.0
        assert summary.std_risk_score == 0.0
        assert summary.risk_buckets == {"low": 0, "medium": 0, "high": 0, "very_high": 0}

    def test_single_decision(self) -> None:
        decisions = [{"risk_score": 30.0, "loan_amount": 5000.0, "decision": "APPROVED", "industry": "finance"}]
        summary = compute_portfolio_summary(decisions)
        assert summary.total_applications == 1
        assert summary.approved_count == 1
        assert summary.total_loan_amount == 5000.0
        assert summary.avg_risk_score == 30.0
        assert summary.median_risk_score == 30.0
        assert summary.std_risk_score == 0.0
        assert summary.approval_rate == 1.0
        assert summary.decline_rate == 0.0

    def test_mixed_decisions(self) -> None:
        decisions = [
            {"risk_score": 10.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"},
            {"risk_score": 40.0, "loan_amount": 2000.0, "decision": "DECLINED", "industry": "b"},
            {"risk_score": 70.0, "loan_amount": 3000.0, "decision": "NEEDS_REVIEW", "industry": "c"},
            {"risk_score": 90.0, "loan_amount": 4000.0, "decision": "APPROVED", "industry": "d"},
        ]
        summary = compute_portfolio_summary(decisions)
        assert summary.total_applications == 4
        assert summary.approved_count == 2
        assert summary.declined_count == 1
        assert summary.manual_review_count == 1
        assert summary.total_loan_amount == 10000.0
        assert summary.approval_rate == 0.5
        assert summary.decline_rate == 0.25
        assert summary.manual_review_rate == 0.25
        assert summary.avg_risk_score == 52.5
        assert summary.median_risk_score == 55.0

    def test_risk_buckets(self) -> None:
        decisions = [
            {"risk_score": 0.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"},
            {"risk_score": 25.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "b"},
            {"risk_score": 50.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "c"},
            {"risk_score": 75.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "d"},
            {"risk_score": 100.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "e"},
        ]
        summary = compute_portfolio_summary(decisions)
        assert summary.risk_buckets["low"] == 2     # 0, 25
        assert summary.risk_buckets["medium"] == 1   # 50
        assert summary.risk_buckets["high"] == 1     # 75
        assert summary.risk_buckets["very_high"] == 1  # 100

    def test_percentiles(self) -> None:
        decisions = [{"risk_score": float(i), "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"} for i in range(1, 101)]
        summary = compute_portfolio_summary(decisions)
        scores = summary.percentile_risk_scores
        assert math.isclose(scores["p5"], 5.95, rel_tol=0.01)
        assert math.isclose(scores["p25"], 25.75, rel_tol=0.01)
        assert math.isclose(scores["p50"], 50.5, rel_tol=0.01)
        assert math.isclose(scores["p75"], 75.25, rel_tol=0.01)
        assert math.isclose(scores["p95"], 95.05, rel_tol=0.01)

    def test_std_dev_single_value(self) -> None:
        decisions = [{"risk_score": 50.0, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"}]
        summary = compute_portfolio_summary(decisions)
        assert summary.std_risk_score == 0.0


class TestStressTest:

    def test_no_shock(self) -> None:
        decisions = [
            {"risk_score": 0.3, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"},
            {"risk_score": 0.5, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "b"},
        ]
        result = run_stress_test(decisions, "none", 0.0)
        assert isinstance(result, PortfolioStressTestResult)
        assert result.scenario_name == "none"
        assert result.shock_factor == 0.0

    def test_shock_increases_loss(self) -> None:
        decisions = [
            {"risk_score": 0.7, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"},
            {"risk_score": 0.6, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "b"},
        ]
        mild = run_stress_test(decisions, "mild", 0.05)
        severe = run_stress_test(decisions, "severe", 0.25)
        assert severe.stressed_loss_rate >= mild.stressed_loss_rate

    def test_empty_decisions(self) -> None:
        result = run_stress_test([], "empty", 0.1)
        assert result.baseline_approval_rate == 0.0
        assert result.stressed_approval_rate == 0.0

    def test_all_declined_no_impact(self) -> None:
        decisions = [
            {"risk_score": 0.9, "loan_amount": 1000.0, "decision": "DECLINED", "industry": "a"},
        ]
        result = run_stress_test(decisions, "test", 0.2)
        assert result.baseline_approval_rate == 0.0

    def test_custom_default_threshold(self) -> None:
        decisions = [
            {"risk_score": 0.7, "loan_amount": 1000.0, "decision": "APPROVED", "industry": "a"},
        ]
        result = run_stress_test(decisions, "test", 0.2, default_threshold=0.85)
        assert result.stressed_loss_rate == 0.0


class TestConcentrationAnalysis:

    def test_hhi_diversified(self) -> None:
        decisions = [
            {"risk_score": 0.1, "loan_amount": 1000.0, "decision": "APPROVED", "industry": f"seg_{i}"}
            for i in range(12)
        ]
        result = concentration_analysis(decisions, "industry")
        assert result["segment"] == "industry"
        assert result["interpretation"] == "diversified"
        assert len(result["top_segments"]) == 5

    def test_hhi_highly_concentrated(self) -> None:
        decisions = [
            {"risk_score": 0.1, "loan_amount": 10000.0, "decision": "APPROVED", "industry": "oil"},
            {"risk_score": 0.2, "loan_amount": 100.0, "decision": "APPROVED", "industry": "gas"},
        ]
        result = concentration_analysis(decisions, "industry")
        assert result["hhi"] > 2500
        assert result["interpretation"] == "highly_concentrated"

    def test_empty_decisions(self) -> None:
        result = concentration_analysis([], "industry")
        assert result["interpretation"] == "no_data"
        assert result["hhi"] == 0.0

    def test_single_segment(self) -> None:
        decisions = [
            {"risk_score": 0.1, "loan_amount": 5000.0, "decision": "APPROVED", "sector": "energy"},
            {"risk_score": 0.2, "loan_amount": 5000.0, "decision": "APPROVED", "sector": "energy"},
        ]
        result = concentration_analysis(decisions, "sector")
        assert result["hhi"] == 10000.0
        assert result["interpretation"] == "highly_concentrated"

    def test_custom_segment_key(self) -> None:
        decisions = [
            {"risk_score": 0.1, "loan_amount": 1000.0, "decision": "APPROVED", "region": "US"},
            {"risk_score": 0.2, "loan_amount": 1000.0, "decision": "APPROVED", "region": "EU"},
        ]
        result = concentration_analysis(decisions, "region")
        assert result["segment"] == "region"
        assert result["hhi"] == 5000.0
