"""Unit tests for pure helper functions in worker/tasks/process_application.py.

These functions have no database, Redis, or Celery dependencies and can be
tested entirely in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.rules import Decision
from services import FailureType, ServiceResult
from worker.tasks.process_application import (
    _decision_user_data,
    _failure_flag,
    _fallback_used_for_reconstructed_result,
    _processing_lock_timeout_seconds,
    _redact_user_data,
    _risk_score_audit_breakdown,
    _service_result_from_external_data,
    _status_for_decision,
)


# ---------------------------------------------------------------------------
# _status_for_decision
# ---------------------------------------------------------------------------


class TestStatusForDecision:
    def _output(self, decision: Decision):
        out = MagicMock()
        out.decision = decision
        return out

    def test_needs_review_returns_manual_review(self):
        assert _status_for_decision(self._output(Decision.NEEDS_REVIEW)) == "MANUAL_REVIEW"

    def test_approve_returns_completed(self):
        assert _status_for_decision(self._output(Decision.APPROVE)) == "COMPLETED"

    def test_decline_returns_completed(self):
        assert _status_for_decision(self._output(Decision.DECLINE)) == "COMPLETED"


# ---------------------------------------------------------------------------
# _failure_flag
# ---------------------------------------------------------------------------


class TestFailureFlag:
    def test_missing_key_returns_none(self):
        assert _failure_flag({}, "credit_bureau") is None

    def test_none_value_returns_none(self):
        assert _failure_flag({"credit_bureau": None}, "credit_bureau") is None

    def test_string_value_converted(self):
        result = _failure_flag({"credit_bureau": "TIMEOUT"}, "credit_bureau")
        assert result == FailureType.TIMEOUT

    def test_failuretype_value_passed_through(self):
        result = _failure_flag({"bank_analyzer": FailureType.SERVICE_DOWN}, "bank_analyzer")
        assert result == FailureType.SERVICE_DOWN

    def test_success_string_returns_success(self):
        result = _failure_flag({"gst_verifier": "SUCCESS"}, "gst_verifier")
        assert result == FailureType.SUCCESS


# ---------------------------------------------------------------------------
# _fallback_used_for_reconstructed_result
# ---------------------------------------------------------------------------


class TestFallbackUsedForReconstructedResult:
    def test_timeout_is_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.TIMEOUT) is True

    def test_service_down_is_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.SERVICE_DOWN) is True

    def test_format_error_is_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.FORMAT_ERROR) is True

    def test_pan_mismatch_is_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.PAN_MISMATCH) is True

    def test_no_record_is_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.NO_RECORD) is True

    def test_none_is_not_fallback(self):
        assert _fallback_used_for_reconstructed_result(None) is False

    def test_success_is_not_fallback(self):
        assert _fallback_used_for_reconstructed_result(FailureType.SUCCESS) is False


# ---------------------------------------------------------------------------
# _processing_lock_timeout_seconds
# ---------------------------------------------------------------------------


class TestProcessingLockTimeoutSeconds:
    def test_default_is_300(self, monkeypatch):
        monkeypatch.delenv("PROCESSING_LOCK_TIMEOUT_SECONDS", raising=False)
        assert _processing_lock_timeout_seconds() == 300

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PROCESSING_LOCK_TIMEOUT_SECONDS", "120")
        assert _processing_lock_timeout_seconds() == 120


# ---------------------------------------------------------------------------
# _redact_user_data
# ---------------------------------------------------------------------------


class TestRedactUserData:
    def test_pan_redacted(self):
        result = _redact_user_data({"pan": "ABCDE1234F"})
        assert result["pan"] == "***REDACTED***"

    def test_name_redacted(self):
        result = _redact_user_data({"name": "Jane Doe"})
        assert result["name"] == "***REDACTED***"

    def test_pan_hash_redacted(self):
        result = _redact_user_data({"pan_hash": "abc123"})
        assert result["pan_hash"] == "***REDACTED***"

    def test_monthly_income_redacted(self):
        result = _redact_user_data({"monthly_income": 100000})
        assert result["monthly_income"] == "***REDACTED***"

    def test_monthly_inflow_redacted(self):
        result = _redact_user_data({"monthly_inflow": 100000})
        assert result["monthly_inflow"] == "***REDACTED***"

    def test_existing_emis_redacted(self):
        result = _redact_user_data({"existing_emis": 20000})
        assert result["existing_emis"] == "***REDACTED***"

    def test_loan_amount_redacted(self):
        result = _redact_user_data({"loan_amount": 500000})
        assert result["loan_amount"] == "***REDACTED***"

    def test_monthly_outflow_redacted(self):
        result = _redact_user_data({"monthly_outflow": 30000})
        assert result["monthly_outflow"] == "***REDACTED***"

    def test_average_balance_redacted(self):
        result = _redact_user_data({"average_balance": 200000})
        assert result["average_balance"] == "***REDACTED***"

    def test_annual_turnover_redacted(self):
        result = _redact_user_data({"annual_turnover": 1500000})
        assert result["annual_turnover"] == "***REDACTED***"

    def test_bank_statement_redacted(self):
        result = _redact_user_data({"bank_statement": [{"amount": 5000}]})
        assert result["bank_statement"] == "***REDACTED***"

    def test_non_pii_key_passes_through(self):
        result = _redact_user_data({"risk_score": 75.0, "decision": "APPROVE"})
        assert result["risk_score"] == pytest.approx(75.0)
        assert result["decision"] == "APPROVE"

    def test_none_input_returns_empty_dict(self):
        assert _redact_user_data(None) == {}

    def test_nested_dict_recursed(self):
        result = _redact_user_data({"external": {"pan": "ABCDE1234F", "score": 700}})
        assert result["external"]["pan"] == "***REDACTED***"
        assert result["external"]["score"] == 700

    def test_list_of_dicts_recursed(self):
        result = _redact_user_data({"items": [{"pan": "ABCDE1234F"}, {"score": 700}]})
        assert result["items"][0]["pan"] == "***REDACTED***"
        assert result["items"][1]["score"] == 700

    def test_list_of_scalars_pass_through(self):
        result = _redact_user_data({"tags": ["a", "b", "c"]})
        assert result["tags"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _decision_user_data
# ---------------------------------------------------------------------------


class TestDecisionUserData:
    def test_required_fields_present(self):
        user_data = {
            "monthly_income": 100000,
            "existing_emis": 25000,
            "loan_amount": 500000,
            "tenure_months": 36,
        }
        result = _decision_user_data(user_data, pan_hash=None)
        assert result["monthly_income"] == 100000
        assert result["existing_emis"] == 25000
        assert result["loan_amount"] == 500000
        assert result["tenure_months"] == 36
        assert "pan_hash" not in result

    def test_pan_hash_included_when_provided(self):
        user_data = {"monthly_income": 100000, "existing_emis": 0}
        result = _decision_user_data(user_data, pan_hash="hashed-pan")
        assert result["pan_hash"] == "hashed-pan"

    def test_missing_optional_fields_default_to_none(self):
        user_data = {"monthly_income": 80000}
        result = _decision_user_data(user_data, pan_hash=None)
        assert result.get("loan_amount") is None
        assert result.get("tenure_months") is None

    def test_no_raw_pan_in_output(self):
        user_data = {"monthly_income": 80000, "pan": "ABCDE1234F", "name": "Jane"}
        result = _decision_user_data(user_data, pan_hash="xyz")
        assert "pan" not in result
        assert "name" not in result


# ---------------------------------------------------------------------------
# _risk_score_audit_breakdown
# ---------------------------------------------------------------------------


class TestRiskScoreAuditBreakdown:
    def _make_service_result(self, failure_type=None):
        r = MagicMock(spec=ServiceResult)
        r.failure_type = failure_type
        return r

    def _make_decision_output(self):
        out = MagicMock()
        out.risk_score = 72.0
        out.confidence = 0.85
        out.data_reliability = 0.9
        out.scoring_strategy = "heuristic"
        out.model_version = None
        return out

    def test_dti_computed(self):
        user_data = {"monthly_income": 100000.0, "existing_emis": 30000.0}
        result = _risk_score_audit_breakdown(
            user_data,
            self._make_decision_output(),
            self._make_service_result(),
            self._make_service_result(),
            self._make_service_result(),
        )
        assert result["dti"] == pytest.approx(0.3)

    def test_dti_none_when_zero_income(self):
        user_data = {"monthly_income": 0, "existing_emis": 30000.0}
        result = _risk_score_audit_breakdown(
            user_data,
            self._make_decision_output(),
            self._make_service_result(),
            self._make_service_result(),
            self._make_service_result(),
        )
        assert result["dti"] is None

    def test_failure_types_extracted(self):
        user_data = {"monthly_income": 100000.0, "existing_emis": 20000.0}
        result = _risk_score_audit_breakdown(
            user_data,
            self._make_decision_output(),
            self._make_service_result(FailureType.TIMEOUT),
            self._make_service_result(None),
            self._make_service_result(FailureType.SERVICE_DOWN),
        )
        assert FailureType.TIMEOUT.value in result["failure_types"]
        assert FailureType.SERVICE_DOWN.value in result["failure_types"]
        assert len(result["failure_types"]) == 2

    def test_components_present(self):
        user_data = {"monthly_income": 100000.0, "existing_emis": 20000.0}
        result = _risk_score_audit_breakdown(
            user_data,
            self._make_decision_output(),
            self._make_service_result(),
            self._make_service_result(),
            self._make_service_result(),
        )
        assert "components" in result
        assert result["components"]["risk_score"] == pytest.approx(72.0)


# ---------------------------------------------------------------------------
# _service_result_from_external_data
# ---------------------------------------------------------------------------


class TestServiceResultFromExternalData:
    def _make_row(self, failure_type=None, response_data=None):
        row = MagicMock()
        row.failure_type = failure_type
        row.response_data = response_data or {}
        return row

    def test_success_when_no_failure(self):
        row = self._make_row(failure_type=None, response_data={"credit_score": 720})
        result = _service_result_from_external_data(row)
        assert result.success is True
        assert result.failure_type is None

    def test_failure_when_failure_type_set(self):
        row = self._make_row(failure_type="TIMEOUT", response_data={})
        result = _service_result_from_external_data(row)
        assert result.success is False
        assert result.failure_type == FailureType.TIMEOUT

    def test_data_set_from_response_data(self):
        row = self._make_row(response_data={"credit_score": 750, "status": "OK"})
        result = _service_result_from_external_data(row)
        assert result.data == {"credit_score": 750, "status": "OK"}

    def test_request_id_extracted(self):
        row = self._make_row(response_data={"request_id": "req-123"})
        result = _service_result_from_external_data(row)
        assert result.request_id == "req-123"

    def test_fallback_used_for_timeout(self):
        row = self._make_row(failure_type="TIMEOUT")
        result = _service_result_from_external_data(row)
        assert result.fallback_used is True

    def test_fallback_not_used_for_success(self):
        row = self._make_row(failure_type=None)
        result = _service_result_from_external_data(row)
        assert result.fallback_used is False

    def test_empty_response_data(self):
        row = self._make_row(response_data=None)
        row.response_data = None
        result = _service_result_from_external_data(row)
        assert result.data is None
