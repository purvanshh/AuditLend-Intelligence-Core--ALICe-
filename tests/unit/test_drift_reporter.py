from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEYS", "test-api-key-for-ci:read-write")
os.environ.setdefault("PII_ENCRYPTION_KEY", "02468ace02468ace02468ace02468ace02468ace02468ace02468ace02468ace")
os.environ.setdefault("PAN_HASH_SALT", "test-salt-for-ci")
os.environ.setdefault("DATABASE_URL", "postgresql://auditlend:auditlend@localhost:5432/auditlend")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from api.main import app
from ml.monitoring.drift_reporter import EVIDENTLY_AVAILABLE, EvidentlyDriftReporter, create_monitoring_dashboard


def _sample_reference_df() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_amount": [10_000.0, 12_000.0, 11_500.0, 13_000.0, 12_500.0] * 20,
        "dti_ratio": [0.10, 0.11, 0.12, 0.10, 0.09] * 20,
    })


def _sample_candidate_shifted_df() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_amount": [45_000.0, 48_000.0, 47_500.0, 49_000.0, 46_500.0] * 20,
        "dti_ratio": [0.10, 0.11, 0.12, 0.10, 0.09] * 20,
    })


def _sample_candidate_same_df() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_amount": [10_000.0, 12_000.0, 11_500.0, 13_000.0, 12_500.0] * 20,
        "dti_ratio": [0.10, 0.11, 0.12, 0.10, 0.09] * 20,
    })


class TestEvidentlyDriftReporterInit:
    def test_initializes_with_reference_data(self) -> None:
        ref = _sample_reference_df()
        reporter = EvidentlyDriftReporter(ref)
        assert reporter.reference_data is ref
        assert len(reporter.reference_data) == 100


class TestComputeDataDrift:
    def test_detects_shifted_features_with_ks_fallback(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)
        result = reporter.compute_data_drift(cand)

        assert "drift_share" in result
        assert "drifted_features" in result
        assert "feature_drift_scores" in result
        assert "dataset_drift" in result
        assert "loan_amount" in result["drifted_features"]
        assert result["drift_share"] > 0

    def test_no_drift_when_distributions_match(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_same_df()
        reporter = EvidentlyDriftReporter(ref)
        result = reporter.compute_data_drift(cand)

        assert len(result["drifted_features"]) == 0
        assert result["drift_share"] == 0.0
        assert result["dataset_drift"] is False

    def test_returns_correct_structure(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)
        result = reporter.compute_data_drift(cand)

        assert isinstance(result["drift_share"], float)
        assert isinstance(result["drifted_features"], list)
        assert isinstance(result["feature_drift_scores"], dict)
        assert isinstance(result["dataset_drift"], bool)

    def test_drift_share_threshold_controls_dataset_drift(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)

        result_high = reporter.compute_data_drift(cand, drift_share=0.8)
        assert result_high["dataset_drift"] is False

        result_low = reporter.compute_data_drift(cand, drift_share=0.0)
        assert result_low["dataset_drift"] is True


class TestGenerateHtmlReport:
    def test_creates_output_file(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result = reporter.generate_html_report(cand, output_path)
            assert Path(output_path).exists()
            assert result == Path(output_path)
            content = Path(output_path).read_text()
            assert len(content) > 0
        finally:
            os.unlink(output_path)

    def test_creates_json_when_evidently_unavailable(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = reporter.generate_html_report(cand, output_path)
            if not EVIDENTLY_AVAILABLE:
                data = json.loads(Path(output_path).read_text())
                assert "drift_share" in data
                assert "drifted_features" in data
        finally:
            os.unlink(output_path)


class TestGenerateTestSuite:
    def test_creates_output_file(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()
        reporter = EvidentlyDriftReporter(ref)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result = reporter.generate_test_suite(cand, output_path)
            assert Path(output_path).exists()
        finally:
            os.unlink(output_path)


class TestCreateMonitoringDashboard:
    def test_creates_report_and_suite_files(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = create_monitoring_dashboard(ref, cand, tmpdir)
            assert len(files) == 2
            for f in files:
                assert Path(f).exists()

    def test_returns_paths_to_generated_files(self) -> None:
        ref = _sample_reference_df()
        cand = _sample_candidate_shifted_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            files = create_monitoring_dashboard(ref, cand, tmpdir)
            assert all(isinstance(f, Path) for f in files)


class TestApiIntegration:
    def test_drift_endpoint_returns_drift_report(self) -> None:
        with TestClient(app) as client:
            ref_data = [{"loan_amount": 10_000.0, "dti_ratio": 0.10}] * 20
            cand_data = [{"loan_amount": 45_000.0, "dti_ratio": 0.10}] * 20

            response = client.post(
                "/api/v1/monitoring/drift",
                json={
                    "reference_data": ref_data,
                    "candidate_data": cand_data,
                    "drift_share_threshold": 0.1,
                },
                headers={"X-API-Key": "test-api-key-for-ci"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "drift_share" in data
            assert "drifted_features" in data
            assert "feature_drift_scores" in data
            assert "dataset_drift" in data

    def test_drift_endpoint_returns_401_without_api_key(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/monitoring/drift",
                json={
                    "reference_data": [],
                    "candidate_data": [],
                },
            )
            assert response.status_code == 401

    def test_reports_list_endpoint(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/monitoring/reports",
                headers={"X-API-Key": "test-api-key-for-ci"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "reports" in data

    def test_generate_report_endpoint(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/monitoring/reports/generate",
                json={},
                headers={"X-API-Key": "test-api-key-for-ci"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "report_path" in data
            assert "test_suite_path" in data
            assert "evidently_available" in data


class TestGracefulDegradation:
    def test_compute_data_drift_works_with_empty_dataframe(self) -> None:
        ref = pd.DataFrame()
        cand = pd.DataFrame()
        reporter = EvidentlyDriftReporter(ref)
        result = reporter.compute_data_drift(cand)
        assert result["drift_share"] == 0.0
        assert result["drifted_features"] == []
        assert result["dataset_drift"] is False

    def test_compute_data_drift_works_with_single_row(self) -> None:
        ref = pd.DataFrame({"loan_amount": [10_000.0]})
        cand = pd.DataFrame({"loan_amount": [15_000.0]})
        reporter = EvidentlyDriftReporter(ref)
        result = reporter.compute_data_drift(cand)
        assert "drift_share" in result
        assert "drifted_features" in result
