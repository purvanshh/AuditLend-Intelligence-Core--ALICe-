"""Unit tests for cli/portfolio.py CLI tool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cli.portfolio import (
    cmd_concentration,
    cmd_html_report,
    cmd_stress_test,
    cmd_summary,
    load_decisions,
    main,
)


def _write_decision_file(path: Path, n: int = 5) -> None:
    decisions = [
        {"risk_score": i * 20.0, "loan_amount": 10000.0 + i * 1000, "decision": "APPROVED", "industry": "tech"}
        for i in range(n)
    ]
    path.write_text(json.dumps(decisions), encoding="utf-8")


def _write_jsonl_file(path: Path, n: int = 5) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            d = {"risk_score": i * 20.0, "loan_amount": 10000.0 + i * 1000, "decision": "APPROVED", "industry": "finance"}
            f.write(json.dumps(d) + "\n")


class FakeNamespace:
    """Mimics argparse.Namespace for testing cmd_* functions."""

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestLoadDecisions:

    def test_load_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            _write_decision_file(path, 3)
            decisions = load_decisions(str(path))
            assert len(decisions) == 3

    def test_load_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            _write_jsonl_file(path, 4)
            decisions = load_decisions(str(path))
            assert len(decisions) == 4

    def test_missing_file_raises(self) -> None:
        import pytest
        with pytest.raises(FileNotFoundError):
            load_decisions("/nonexistent/path.json")


class TestCmdSummary:

    def test_summary_with_output(self, capsys: object) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            output_path = Path(tmpdir) / "out.md"
            _write_decision_file(input_path, 10)
            args = FakeNamespace(input=str(input_path), output=str(output_path))
            cmd_summary(args)
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "# Portfolio Risk Report" in content

    def test_summary_no_output(self, capsys: object) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            _write_decision_file(input_path, 3)
            args = FakeNamespace(input=str(input_path), output=None)
            cmd_summary(args)


class TestCmdStressTest:

    def test_stress_test_with_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            output_path = Path(tmpdir) / "stress.md"
            _write_decision_file(input_path, 10)
            args = FakeNamespace(input=str(input_path), scenario="recession", shock=0.15, output=str(output_path))
            cmd_stress_test(args)
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "# Portfolio Stress Test Report" in content

    def test_stress_test_no_output(self, capsys: object) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            _write_decision_file(input_path, 5)
            args = FakeNamespace(input=str(input_path), scenario="test", shock=0.1, output=None)
            cmd_stress_test(args)


class TestCmdConcentration:

    def test_concentration_with_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            output_path = Path(tmpdir) / "conc.json"
            _write_decision_file(input_path, 10)
            args = FakeNamespace(input=str(input_path), segment="industry", output=str(output_path))
            cmd_concentration(args)
            assert output_path.exists()
            data = json.loads(output_path.read_text(encoding="utf-8"))
            assert "hhi" in data
            assert data["segment"] == "industry"

    def test_concentration_no_output(self, capsys: object) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            _write_decision_file(input_path, 5)
            args = FakeNamespace(input=str(input_path), segment="industry", output=None)
            cmd_concentration(args)


class TestCmdHtmlReport:

    def test_html_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            output_path = Path(tmpdir) / "report.html"
            _write_decision_file(input_path, 10)
            args = FakeNamespace(input=str(input_path), output=str(output_path))
            cmd_html_report(args)
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content


class TestMain:

    def test_summary_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.json"
            output_path = Path(tmpdir) / "out.md"
            _write_decision_file(input_path, 5)
            import sys
            old_argv = sys.argv
            try:
                sys.argv = ["portfolio.py", "summary", "--input", str(input_path), "--output", str(output_path)]
                main()
                assert output_path.exists()
            finally:
                sys.argv = old_argv

    def test_missing_input_file(self) -> None:
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["portfolio.py", "summary", "--input", "/nonexistent/path.json"]
            import pytest
            with pytest.raises(FileNotFoundError):
                main()
        finally:
            sys.argv = old_argv
