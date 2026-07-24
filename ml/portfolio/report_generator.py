from __future__ import annotations

import html as html_module
from pathlib import Path
from typing import Any

from ml.portfolio.risk_aggregator import PortfolioSummary, PortfolioStressTestResult


def _risk_bar(ratio: float, width: int = 30) -> str:
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def generate_portfolio_report(
    summary: PortfolioSummary,
    output_path: str | Path,
) -> str:
    lines: list[str] = []
    lines.append("# Portfolio Risk Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total Applications | {summary.total_applications} |")
    lines.append(f"| Approved | {summary.approved_count} |")
    lines.append(f"| Declined | {summary.declined_count} |")
    lines.append(f"| Manual Review | {summary.manual_review_count} |")
    lines.append(f"| Total Loan Amount | {summary.total_loan_amount:,.2f} |")
    lines.append(f"| Average Risk Score | {summary.avg_risk_score:.2f} |")
    lines.append(f"| Median Risk Score | {summary.median_risk_score:.2f} |")
    lines.append(f"| Std Dev Risk Score | {summary.std_risk_score:.2f} |")
    lines.append(f"| Approval Rate | {summary.approval_rate:.2%} |")
    lines.append(f"| Decline Rate | {summary.decline_rate:.2%} |")
    lines.append(f"| Manual Review Rate | {summary.manual_review_rate:.2%} |")
    lines.append("")

    if summary.percentile_risk_scores:
        lines.append("## Risk Score Distribution")
        lines.append("")
        lines.append("| Percentile | Score |")
        lines.append("|---|---:|")
        for key in ("p5", "p25", "p50", "p75", "p95"):
            val = summary.percentile_risk_scores.get(key, 0.0)
            lines.append(f"| {key} | {val:.2f} |")
        lines.append("")

    lines.append("## Risk Bucket Distribution")
    lines.append("")
    total = summary.total_applications or 1
    bucket_order = [("low", "🟢 Low"), ("medium", "🟡 Medium"), ("high", "🟠 High"), ("very_high", "🔴 Very High")]
    for key, label in bucket_order:
        count = summary.risk_buckets.get(key, 0)
        ratio = count / total
        bar = _risk_bar(ratio)
        lines.append(f"| {label} | {bar} | {count:>4} ({ratio:.1%}) |")
    lines.append("")

    lines.append("## Approval Breakdown")
    lines.append("")
    lines.append("| Decision | Count | Percentage |")
    lines.append("|---|---|---:|")
    lines.append(f"| ✅ Approved | {summary.approved_count} | {summary.approval_rate:.1%} |")
    lines.append(f"| ❌ Declined | {summary.declined_count} | {summary.decline_rate:.1%} |")
    lines.append(f"| 🔍 Manual Review | {summary.manual_review_count} | {summary.manual_review_rate:.1%} |")
    lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def generate_stress_test_report(
    results: list[PortfolioStressTestResult],
    output_path: str | Path,
) -> str:
    lines: list[str] = []
    lines.append("# Portfolio Stress Test Report")
    lines.append("")

    for i, result in enumerate(results, 1):
        lines.append(f"## Scenario {i}: {result.scenario_name}")
        lines.append("")
        lines.append(f"- **Shock Factor:** +{result.shock_factor:.0%}")
        lines.append(f"- **Baseline Approval Rate:** {result.baseline_approval_rate:.2%}")
        lines.append(f"- **Stressed Approval Rate:** {result.stressed_approval_rate:.2%}")
        lines.append(f"- **Baseline Loss Rate:** {result.baseline_loss_rate:.2%}")
        lines.append(f"- **Stressed Loss Rate:** {result.stressed_loss_rate:.2%}")
        lines.append(f"- **Additional Loss:** {result.additional_loss_pct:+.2f}%")
        lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def portfolio_report_to_html(
    summary: PortfolioSummary,
    output_path: str | Path,
) -> str:
    total = summary.total_applications or 1
    bucket_colors = {"low": "#4CAF50", "medium": "#FFC107", "high": "#FF9800", "very_high": "#F44336"}

    def bar_html(label: str, count: int, color: str) -> str:
        pct = count / total * 100
        return f"""
        <div style="margin:8px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span>{label}</span>
                <span>{count} ({pct:.1f}%)</span>
            </div>
            <div style="background:#e0e0e0;border-radius:4px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:24px;background:{color};border-radius:4px;"></div>
            </div>
        </div>"""

    bucket_html = "".join(
        bar_html(key.capitalize(), summary.risk_buckets.get(key, 0), bucket_colors[key])
        for key in ("low", "medium", "high", "very_high")
    )

    def percentile_rows() -> str:
        rows = ""
        for key in ("p5", "p25", "p50", "p75", "p95"):
            val = summary.percentile_risk_scores.get(key, 0.0)
            rows += f"<tr><td>{key}</td><td style='text-align:right'>{val:.2f}</td></tr>\n"
        return rows

    def decision_rows() -> str:
        return f"""
        <tr><td>Approved</td><td style='text-align:right'>{summary.approved_count}</td><td style='text-align:right'>{summary.approval_rate:.1%}</td></tr>
        <tr><td>Declined</td><td style='text-align:right'>{summary.declined_count}</td><td style='text-align:right'>{summary.decline_rate:.1%}</td></tr>
        <tr><td>Manual Review</td><td style='text-align:right'>{summary.manual_review_count}</td><td style='text-align:right'>{summary.manual_review_rate:.1%}</td></tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Risk Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; color: #333; }}
h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
h2 {{ color: #283593; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 600px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #f5f5f5; }}
.container {{ max-width: 800px; margin: 0 auto; }}
.card {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 16px 0; }}
</style>
</head>
<body>
<div class="container">
<h1>Portfolio Risk Report</h1>

<div class="card">
<h2>Summary</h2>
<table>
<tr><th>Metric</th><th style='text-align:right'>Value</th></tr>
<tr><td>Total Applications</td><td style='text-align:right'>{summary.total_applications}</td></tr>
<tr><td>Approved</td><td style='text-align:right'>{summary.approved_count}</td></tr>
<tr><td>Declined</td><td style='text-align:right'>{summary.declined_count}</td></tr>
<tr><td>Manual Review</td><td style='text-align:right'>{summary.manual_review_count}</td></tr>
<tr><td>Total Loan Amount</td><td style='text-align:right'>{summary.total_loan_amount:,.2f}</td></tr>
<tr><td>Average Risk Score</td><td style='text-align:right'>{summary.avg_risk_score:.2f}</td></tr>
<tr><td>Median Risk Score</td><td style='text-align:right'>{summary.median_risk_score:.2f}</td></tr>
<tr><td>Std Dev Risk Score</td><td style='text-align:right'>{summary.std_risk_score:.2f}</td></tr>
<tr><td>Approval Rate</td><td style='text-align:right'>{summary.approval_rate:.2%}</td></tr>
<tr><td>Decline Rate</td><td style='text-align:right'>{summary.decline_rate:.2%}</td></tr>
<tr><td>Manual Review Rate</td><td style='text-align:right'>{summary.manual_review_rate:.2%}</td></tr>
</table>
</div>

<div class="card">
<h2>Risk Buckets</h2>
{bucket_html}
</div>

<div class="card">
<h2>Percentile Distribution</h2>
<table>
<tr><th>Percentile</th><th style='text-align:right'>Score</th></tr>
{percentile_rows()}
</table>
</div>

<div class="card">
<h2>Approval Breakdown</h2>
<table>
<tr><th>Decision</th><th style='text-align:right'>Count</th><th style='text-align:right'>Percentage</th></tr>
{decision_rows()}
</table>
</div>

</div>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")
    return str(path)
