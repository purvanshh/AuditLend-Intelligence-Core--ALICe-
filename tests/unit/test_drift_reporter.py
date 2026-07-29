"""Tests for EvidentlyDriftReporter (ml/monitoring/drift_reporter.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.monitoring.drift_reporter import (
    EVIDENTLY_AVAILABLE,
    EvidentlyDriftReporter,
    create_monitoring_dashboard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_df() -> pd.DataFrame:
    return pd.DataFrame({
        "dti": [0.2, 0.3, 0.4, 0.35, 0.25],
        "income": [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
        "credit_score": [700, 720, 680, 710, 690],
    })


def _cand_df() -> pd.DataFrame:
    return pd.DataFrame({
        "dti": [0.3, 0.4, 0.5, 0.45, 0.35],
        "income": [45000.0, 55000.0, 65000.0, 50000.0, 60000.0],
        "credit_score": [680, 700, 660, 690, 670],
    })


# ---------------------------------------------------------------------------
# _compute_with_ks_fallback (Evidently not available)
# ---------------------------------------------------------------------------


class TestKsFallback:
    def test_returns_dict_with_required_keys(self):
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df())

        assert "drift_share" in result
        assert "drifted_features" in result
        assert "feature_drift_scores" in result
        assert "dataset_drift" in result

    def test_drift_share_in_range(self):
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df())

        assert 0.0 <= result["drift_share"] <= 1.0

    def test_dataset_drift_is_bool(self):
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df())

        assert isinstance(result["dataset_drift"], bool)

    def test_drifted_features_is_list(self):
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df())

        assert isinstance(result["drifted_features"], list)

    def test_feature_drift_scores_contains_expected_cols(self):
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df())

        scores = result["feature_drift_scores"]
        # KS fallback returns scores for columns that appear in both ref and candidate
        assert isinstance(scores, dict)

    def test_threshold_respected_for_dataset_drift(self):
        # With a very high threshold, dataset_drift should be False even with drift
        reporter = EvidentlyDriftReporter(_ref_df())
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result = reporter.compute_data_drift(_cand_df(), drift_share=1.0)

        assert result["dataset_drift"] is False


# ---------------------------------------------------------------------------
# generate_html_report (no Evidently → JSON fallback)
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    def test_creates_file_when_evidently_unavailable(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "report.json"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result_path = reporter.generate_html_report(_cand_df(), output)

        assert result_path == output
        assert output.exists()

    def test_json_output_is_valid_when_evidently_unavailable(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "report.json"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            reporter.generate_html_report(_cand_df(), output)

        data = json.loads(output.read_text())
        assert "drift_share" in data

    def test_creates_parent_dirs(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "subdir" / "nested" / "report.json"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            reporter.generate_html_report(_cand_df(), output)

        assert output.exists()

    def test_html_report_with_evidently_mocked(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "report.html"

        mock_report = MagicMock()
        import ml.monitoring.drift_reporter as mod
        original_report = getattr(mod, "Report", None)
        original_table = getattr(mod, "DataDriftTable", None)
        mod.Report = MagicMock(return_value=mock_report)
        mod.DataDriftTable = MagicMock()
        try:
            with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", True):
                reporter.generate_html_report(_cand_df(), output)
        finally:
            if original_report is not None:
                mod.Report = original_report
            elif hasattr(mod, "Report"):
                del mod.Report
            if original_table is not None:
                mod.DataDriftTable = original_table
            elif hasattr(mod, "DataDriftTable"):
                del mod.DataDriftTable

        mock_report.run.assert_called_once()
        mock_report.save_html.assert_called_once_with(str(output))


# ---------------------------------------------------------------------------
# generate_test_suite
# ---------------------------------------------------------------------------


class TestGenerateTestSuite:
    def test_creates_file_when_evidently_unavailable(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "suite.json"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            result_path = reporter.generate_test_suite(_cand_df(), output)

        assert result_path == output
        assert output.exists()

    def test_suite_json_valid(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "suite.json"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            reporter.generate_test_suite(_cand_df(), output)

        data = json.loads(output.read_text())
        assert "drift_share" in data

    def test_test_suite_with_evidently_mocked(self, tmp_path):
        reporter = EvidentlyDriftReporter(_ref_df())
        output = tmp_path / "suite.html"

        mock_suite = MagicMock()
        import ml.monitoring.drift_reporter as mod
        original_suite = getattr(mod, "TestSuite", None)
        original_drift_test = getattr(mod, "TestFeatureValueDrift", None)
        mod.TestSuite = MagicMock(return_value=mock_suite)
        mod.TestFeatureValueDrift = MagicMock()
        try:
            with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", True):
                reporter.generate_test_suite(_cand_df(), output)
        finally:
            if original_suite is not None:
                mod.TestSuite = original_suite
            elif hasattr(mod, "TestSuite"):
                del mod.TestSuite
            if original_drift_test is not None:
                mod.TestFeatureValueDrift = original_drift_test
            elif hasattr(mod, "TestFeatureValueDrift"):
                del mod.TestFeatureValueDrift

        mock_suite.run.assert_called_once()
        mock_suite.save_html.assert_called_once_with(str(output))


# ---------------------------------------------------------------------------
# create_monitoring_dashboard
# ---------------------------------------------------------------------------


class TestCreateMonitoringDashboard:
    def test_returns_two_paths(self, tmp_path):
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            paths = create_monitoring_dashboard(_ref_df(), _cand_df(), tmp_path)

        assert len(paths) == 2
        for p in paths:
            assert isinstance(p, Path)
            assert p.exists()

    def test_output_in_specified_dir(self, tmp_path):
        output_dir = tmp_path / "monitoring"
        with patch("ml.monitoring.drift_reporter.EVIDENTLY_AVAILABLE", False):
            paths = create_monitoring_dashboard(_ref_df(), _cand_df(), output_dir)

        for p in paths:
            assert str(output_dir) in str(p)
