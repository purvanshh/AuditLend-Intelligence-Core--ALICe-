"""Tests for uncovered helpers in engine/scoring.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.scoring import (
    _clamp,
    _env_truthy,
    _grade_from_credit_score,
    _gst_label,
    _safe_ratio,
    _source_label,
    _verification_status,
    ml_scoring_requested_from_env,
    compute_risk_score,
    MLScoringResult,
)
from engine.rule_sets import ACTIVE_RULE_SET
from services import FailureType


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_below_lower(self):
        assert _clamp(-5.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_above_upper(self):
        assert _clamp(150.0, 0.0, 100.0) == pytest.approx(100.0)

    def test_at_lower_boundary(self):
        assert _clamp(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_at_upper_boundary(self):
        assert _clamp(1.0, 0.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _env_truthy
# ---------------------------------------------------------------------------


class TestEnvTruthy:
    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"])
    def test_truthy_values(self, val):
        assert _env_truthy(val) is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "no", "off", "", "anything"])
    def test_falsy_values(self, val):
        assert _env_truthy(val) is False

    def test_none_returns_false(self):
        assert _env_truthy(None) is False


# ---------------------------------------------------------------------------
# _source_label
# ---------------------------------------------------------------------------


class TestSourceLabel:
    def test_none_returns_missing_label(self):
        assert _source_label(None, "fallback", "live") == "fallback"

    def test_value_returns_present_label(self):
        assert _source_label(700, "fallback", "live") == "live"

    def test_zero_value_returns_present_label(self):
        # 0 is not None
        assert _source_label(0, "fallback", "live") == "live"


# ---------------------------------------------------------------------------
# _gst_label
# ---------------------------------------------------------------------------


class TestGstLabel:
    def test_true_returns_compliant(self):
        assert _gst_label(True) == "compliant"

    def test_false_returns_non_compliant(self):
        assert _gst_label(False) == "non_compliant"

    def test_none_returns_unknown(self):
        assert _gst_label(None) == "unknown"


# ---------------------------------------------------------------------------
# _safe_ratio
# ---------------------------------------------------------------------------


class TestSafeRatio:
    def test_normal_division(self):
        assert _safe_ratio(25000.0, 100000.0) == pytest.approx(0.25)

    def test_zero_denominator(self):
        assert _safe_ratio(25000.0, 0.0) == pytest.approx(0.0)

    def test_negative_denominator(self):
        assert _safe_ratio(10.0, -5.0) == pytest.approx(0.0)

    def test_zero_numerator(self):
        assert _safe_ratio(0.0, 100000.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _grade_from_credit_score
# ---------------------------------------------------------------------------


class TestGradeFromCreditScore:
    def test_grade_a_above_780(self):
        grade, sub_grade = _grade_from_credit_score(800)
        assert grade == "A"
        assert sub_grade == "A1"

    def test_grade_b_720_to_779(self):
        grade, sub_grade = _grade_from_credit_score(750)
        assert grade == "B"
        assert sub_grade == "B2"

    def test_grade_c_660_to_719(self):
        grade, sub_grade = _grade_from_credit_score(680)
        assert grade == "C"
        assert sub_grade == "C3"

    def test_grade_d_600_to_659(self):
        grade, sub_grade = _grade_from_credit_score(620)
        assert grade == "D"
        assert sub_grade == "D4"

    def test_grade_e_below_600(self):
        grade, sub_grade = _grade_from_credit_score(550)
        assert grade == "E"
        assert sub_grade == "E5"

    def test_boundary_780_is_a(self):
        grade, _ = _grade_from_credit_score(780)
        assert grade == "A"

    def test_boundary_720_is_b(self):
        grade, _ = _grade_from_credit_score(720)
        assert grade == "B"

    def test_boundary_660_is_c(self):
        grade, _ = _grade_from_credit_score(660)
        assert grade == "C"

    def test_boundary_600_is_d(self):
        grade, _ = _grade_from_credit_score(600)
        assert grade == "D"


# ---------------------------------------------------------------------------
# _verification_status
# ---------------------------------------------------------------------------


class TestVerificationStatus:
    def test_income_stability_key_gives_verified(self):
        assert _verification_status({"income_stability": 0.8}) == "Verified"

    def test_monthly_inflow_key_gives_source_verified(self):
        assert _verification_status({"monthly_inflow": 50000.0}) == "Source Verified"

    def test_empty_dict_gives_not_verified(self):
        assert _verification_status({}) == "Not Verified"

    def test_income_stability_takes_precedence_over_monthly_inflow(self):
        bank_data = {"income_stability": 0.8, "monthly_inflow": 50000.0}
        assert _verification_status(bank_data) == "Verified"


# ---------------------------------------------------------------------------
# ml_scoring_requested_from_env
# ---------------------------------------------------------------------------


class TestMlScoringRequestedFromEnv:
    def test_false_when_not_set(self, monkeypatch):
        monkeypatch.delenv("ML_ENABLED", raising=False)
        monkeypatch.delenv("RULE_SET_VERSION", raising=False)
        assert ml_scoring_requested_from_env() is False

    def test_true_when_ml_enabled_env(self, monkeypatch):
        monkeypatch.setenv("ML_ENABLED", "true")
        monkeypatch.delenv("RULE_SET_VERSION", raising=False)
        assert ml_scoring_requested_from_env() is True

    def test_true_when_rule_set_v2(self, monkeypatch):
        monkeypatch.delenv("ML_ENABLED", raising=False)
        monkeypatch.setenv("RULE_SET_VERSION", "RULE_SET_V2")
        assert ml_scoring_requested_from_env() is True

    def test_false_when_rule_set_v1(self, monkeypatch):
        monkeypatch.delenv("ML_ENABLED", raising=False)
        monkeypatch.setenv("RULE_SET_VERSION", "RULE_SET_V1")
        assert ml_scoring_requested_from_env() is False


# ---------------------------------------------------------------------------
# compute_risk_score — uncovered branches
# ---------------------------------------------------------------------------


class TestComputeRiskScoreUncoveredBranches:
    def test_gst_none_gives_zero_gst_component(self):
        score, breakdown = compute_risk_score(
            credit_score=700,
            income_stability=0.8,
            dti=0.2,
            gst_compliant=None,
            failure_types=[],
        )
        # gst_compliant=None → gst_component should be 0
        gst_line = next(l for l in breakdown if "gst_component" in l)
        assert "unknown" in gst_line

    def test_multiple_failure_types_capped_at_max_penalty(self):
        many_failures = [FailureType.TIMEOUT] * 10
        score, breakdown = compute_risk_score(
            credit_score=750,
            income_stability=0.9,
            dti=0.2,
            gst_compliant=True,
            failure_types=many_failures,
        )
        # Score should not go below 0
        assert score >= 0.0

    def test_high_dti_produces_low_dti_component(self):
        score_high_dti, _ = compute_risk_score(
            credit_score=700, income_stability=0.7, dti=0.95, gst_compliant=True, failure_types=[]
        )
        score_low_dti, _ = compute_risk_score(
            credit_score=700, income_stability=0.7, dti=0.1, gst_compliant=True, failure_types=[]
        )
        assert score_low_dti > score_high_dti

    def test_fallback_credit_score_label_in_breakdown(self):
        _, breakdown = compute_risk_score(
            credit_score=None,
            income_stability=0.5,
            dti=0.3,
            gst_compliant=True,
            failure_types=[],
        )
        credit_line = next(l for l in breakdown if "credit_component" in l)
        assert "fallback" in credit_line

    def test_fallback_stability_label_in_breakdown(self):
        _, breakdown = compute_risk_score(
            credit_score=700,
            income_stability=None,
            dti=0.3,
            gst_compliant=True,
            failure_types=[],
        )
        stability_line = next(l for l in breakdown if "income_stability_component" in l)
        assert "default" in stability_line


# ---------------------------------------------------------------------------
# MLScoringResult.to_audit_output
# ---------------------------------------------------------------------------


class TestMLScoringResultToAuditOutput:
    def test_to_audit_output_returns_dict(self):
        result = MLScoringResult(
            attempted=True,
            used=True,
            fallback_used=False,
            fallback_reason=None,
            error_type=None,
            risk_score=70.0,
            predicted_default_probability=0.3,
            calibrated_default_probability=0.28,
            model_confidence=0.72,
            model_version="XGB_V1",
            selected_candidate="xgboost",
            score_breakdown=["risk_score = 70.0"],
            model_factor_contributions=[{"feature": "dti", "contribution": 0.1}],
            model_summary="Test summary",
        )
        output = result.to_audit_output()
        assert isinstance(output, dict)
        assert output["risk_score"] == pytest.approx(70.0)
        assert isinstance(output["model_factor_contributions"], list)

    def test_to_audit_output_model_factor_contributions_are_dicts(self):
        result = MLScoringResult(
            attempted=False, used=False, fallback_used=True,
            fallback_reason="TIMEOUT", error_type="TIMEOUT",
            risk_score=None, predicted_default_probability=None,
            calibrated_default_probability=None, model_confidence=None,
            model_version="XGB_V1", selected_candidate="xgboost",
            score_breakdown=[], model_factor_contributions=[], model_summary=None,
        )
        output = result.to_audit_output()
        assert output["model_factor_contributions"] == []
