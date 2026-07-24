"""CLI for AuditLend portfolio analysis.

Usage:
    python -m cli.portfolio summary --input decisions.json [--output report.md]
    python -m cli.portfolio stress-test --input decisions.json --scenario "recession" --shock 0.15
    python -m cli.portfolio concentration --input decisions.json --segment industry
    python -m cli.portfolio html-report --input decisions.json --output report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ml.portfolio.risk_aggregator import (
    compute_portfolio_summary,
    concentration_analysis,
    run_stress_test,
)
from ml.portfolio.report_generator import (
    generate_portfolio_report,
    generate_stress_test_report,
    portfolio_report_to_html,
)


def load_decisions(path: str) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return data


def cmd_summary(args: argparse.Namespace) -> None:
    decisions = load_decisions(args.input)
    summary = compute_portfolio_summary(decisions)
    if args.output:
        path = generate_portfolio_report(summary, args.output)
        print(f"Report written to {path}")
    else:
        import pprint
        pprint.pprint(summary)


def cmd_stress_test(args: argparse.Namespace) -> None:
    decisions = load_decisions(args.input)
    result = run_stress_test(decisions, scenario_name=args.scenario, shock_factor=args.shock)
    if args.output:
        path = generate_stress_test_report([result], args.output)
        print(f"Report written to {path}")
    else:
        import pprint
        pprint.pprint(result)


def cmd_concentration(args: argparse.Namespace) -> None:
    decisions = load_decisions(args.input)
    result = concentration_analysis(decisions, segment_key=args.segment)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Analysis written to {args.output}")
    else:
        import pprint
        pprint.pprint(result)


def cmd_html_report(args: argparse.Namespace) -> None:
    decisions = load_decisions(args.input)
    summary = compute_portfolio_summary(decisions)
    path = portfolio_report_to_html(summary, args.output)
    print(f"HTML report written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AuditLend Portfolio Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_summary = subparsers.add_parser("summary", help="Compute portfolio summary")
    p_summary.add_argument("--input", required=True, help="Path to decisions JSON file")
    p_summary.add_argument("--output", default=None, help="Output report path (optional)")

    p_stress = subparsers.add_parser("stress-test", help="Run stress test on portfolio")
    p_stress.add_argument("--input", required=True, help="Path to decisions JSON file")
    p_stress.add_argument("--scenario", required=True, help="Scenario name")
    p_stress.add_argument("--shock", required=True, type=float, help="Risk shock factor (e.g. 0.15)")
    p_stress.add_argument("--output", default=None, help="Output report path (optional)")

    p_conc = subparsers.add_parser("concentration", help="Concentration analysis")
    p_conc.add_argument("--input", required=True, help="Path to decisions JSON file")
    p_conc.add_argument("--segment", default="industry", help="Segment key for concentration analysis")
    p_conc.add_argument("--output", default=None, help="Output path (optional)")

    p_html = subparsers.add_parser("html-report", help="Generate HTML portfolio report")
    p_html.add_argument("--input", required=True, help="Path to decisions JSON file")
    p_html.add_argument("--output", required=True, help="Output HTML file path")

    args = parser.parse_args()

    commands = {
        "summary": cmd_summary,
        "stress-test": cmd_stress_test,
        "concentration": cmd_concentration,
        "html-report": cmd_html_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
