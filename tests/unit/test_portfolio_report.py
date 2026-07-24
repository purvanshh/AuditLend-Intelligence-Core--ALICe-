"""Unit tests for ml/portfolio/report_generator.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ml.portfolio.risk_aggregator import PortfolioSummary, PortfolioStressTestResult
from ml.portfolio.report_generator import (
    generate_portfolio_report,
    generate_stress_test_report,
    portfolio_report_to_html,
)


def _sample_summary() -> PortfolioSummary:
    return PortfolioSummary(
        total_applications=100,
        approved_count=60,
        declined_count=25,
        manual_review_count=15,
        total_loan_amount=2_500_000.0,
        avg_risk_score=42.5,
        median_risk_score=40.0,
        std_risk_score=15.3,
        percentile_risk_scores={"p5": 10.0, "p25": 28.0, "p50": 40.0, "p75": 55.0, "p95": 72.0},
        approval_rate=0.6,
        decline_rate=0.25,
        manual_review_rate=0.15,
        risk_buckets={"low": 30, "medium": 40, "high": 20, "very_high": 10},
    )


class TestGeneratePortfolioReport:

    def test_creates_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            result = generate_portfolio_report(_sample_summary(), path)
            assert result == str(path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert content.startswith("# Portfolio Risk Report")

    def test_contains_summary_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            generate_portfolio_report(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "| Metric | Value |" in content
            assert "| Total Applications | 100 |" in content

    def test_contains_risk_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            generate_portfolio_report(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "## Risk Bucket Distribution" in content
            assert "Low" in content or "low" in content

    def test_contains_approval_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            generate_portfolio_report(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "## Approval Breakdown" in content
            assert "Approved" in content

    def test_empty_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.md"
            summary = PortfolioSummary(total_applications=0)
            generate_portfolio_report(summary, path)
            content = path.read_text(encoding="utf-8")
            assert content.startswith("# Portfolio Risk Report")


class TestGenerateStressTestReport:

    def test_creates_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stress.md"
            results = [
                PortfolioStressTestResult(
                    scenario_name="Recession",
                    shock_factor=0.15,
                    baseline_approval_rate=0.8,
                    stressed_approval_rate=0.6,
                    baseline_loss_rate=0.05,
                    stressed_loss_rate=0.15,
                    additional_loss_pct=200.0,
                ),
            ]
            result = generate_stress_test_report(results, path)
            assert result == str(path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert content.startswith("# Portfolio Stress Test Report")

    def test_multiple_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stress.md"
            results = [
                PortfolioStressTestResult("A", 0.1, 0.8, 0.7, 0.05, 0.1, 100.0),
                PortfolioStressTestResult("B", 0.2, 0.8, 0.5, 0.05, 0.2, 300.0),
            ]
            generate_stress_test_report(results, path)
            content = path.read_text(encoding="utf-8")
            assert "Scenario 1: A" in content
            assert "Scenario 2: B" in content


class TestPortfolioReportToHtml:

    def test_creates_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            result = portfolio_report_to_html(_sample_summary(), path)
            assert result == str(path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content

    def test_contains_inline_css(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            portfolio_report_to_html(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "<style>" in content

    def test_contains_risk_buckets_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            portfolio_report_to_html(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "Low" in content
            assert "Medium" in content
            assert "High" in content
            assert "Very High" in content or "Very_High" not in content
            for color in ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]:
                assert color in content

    def test_contains_summary_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            portfolio_report_to_html(_sample_summary(), path)
            content = path.read_text(encoding="utf-8")
            assert "2,500,000" in content
            assert "42.5" in content
