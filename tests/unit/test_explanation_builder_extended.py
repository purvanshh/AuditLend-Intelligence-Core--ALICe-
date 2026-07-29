"""Extended tests for engine/explanation_builder.py — covers uncovered helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.explanation_builder import (
    _append_drift_summary,
    _append_model_summary,
    _append_warning_and_model_summary,
    _confidence_text,
    _degradation_sentence,
    _factor_objects,
    _model_details,
    _model_payload_from_decision,
    _model_summary_from_contributions,
    _serialise_model_fragment,
    _timeline_entry,
)


# ---------------------------------------------------------------------------
# _confidence_text
# ---------------------------------------------------------------------------


class TestConfidenceText:
    def test_none_returns_unknown(self):
        assert _confidence_text(None) == "unknown"

    def test_float_formatted_two_decimal(self):
        assert _confidence_text(0.85) == "0.85"

    def test_zero(self):
        assert _confidence_text(0.0) == "0.00"

    def test_one(self):
        assert _confidence_text(1.0) == "1.00"


# ---------------------------------------------------------------------------
# _degradation_sentence
# ---------------------------------------------------------------------------


class TestDegradationSentence:
    def test_empty_returns_no_degradation(self):
        result = _degradation_sentence([])
        assert "No external data degradation" in result

    def test_single_entry(self):
        entry = MagicMock()
        entry.step = "CREDIT_BUREAU_FETCH"
        entry.error_type = "TIMEOUT"
        result = _degradation_sentence([entry])
        assert "TIMEOUT" in result
        assert "Credit Bureau" in result

    def test_no_error_type_shows_fallback(self):
        entry = MagicMock()
        entry.step = "BANK_ANALYZER_FETCH"
        entry.error_type = None
        result = _degradation_sentence([entry])
        assert "fallback" in result


# ---------------------------------------------------------------------------
# _factor_objects
# ---------------------------------------------------------------------------


class TestFactorObjects:
    def test_parses_factor_with_source(self):
        factors = ["credit_component(live) = 25.00/40.00"]
        result = _factor_objects(factors)
        assert len(result) == 1
        assert result[0]["value"] == "25.00/40.00"
        assert result[0]["status"] == "live"

    def test_parses_factor_without_source(self):
        factors = ["dti_component = 18.00/20.00"]
        result = _factor_objects(factors)
        assert len(result) == 1
        assert result[0]["status"] == "derived"

    def test_skips_factor_without_equals(self):
        factors = ["risk_score: 72"]
        result = _factor_objects(factors)
        assert result == []

    def test_empty_input(self):
        assert _factor_objects([]) == []

    def test_multiple_factors(self):
        factors = [
            "credit_component(live) = 25.00",
            "dti_component = 18.00",
            "gst_component(compliant) = 10.00",
        ]
        result = _factor_objects(factors)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _timeline_entry
# ---------------------------------------------------------------------------


class TestTimelineEntry:
    def test_error_type_as_status(self):
        entry = MagicMock()
        entry.step = "CREDIT_BUREAU_FETCH"
        entry.error_type = "TIMEOUT"
        entry.output_snapshot = {}
        entry.created_at = "2026-07-29T10:00:00"
        result = _timeline_entry(entry)
        assert result["status"] == "TIMEOUT"
        assert result["step"] == "CREDIT_BUREAU_FETCH"

    def test_no_error_uses_output_decision(self):
        entry = MagicMock()
        entry.step = "DECISION_CALCULATION"
        entry.error_type = None
        entry.output_snapshot = {"decision": "APPROVE"}
        entry.created_at = "2026-07-29T10:00:00"
        result = _timeline_entry(entry)
        assert result["status"] == "APPROVE"

    def test_no_error_no_decision_uses_success(self):
        entry = MagicMock()
        entry.step = "PROCESSING_STARTED"
        entry.error_type = None
        entry.output_snapshot = {}
        entry.created_at = "2026-07-29T10:00:00"
        result = _timeline_entry(entry)
        assert result["status"] == "SUCCESS"


# ---------------------------------------------------------------------------
# _model_payload_from_decision
# ---------------------------------------------------------------------------


class TestModelPayloadFromDecision:
    def test_empty_payload_returns_empty_contributions(self):
        result = _model_payload_from_decision({})
        assert result["model_factor_contributions"] == []
        assert result["model_version"] is None

    def test_contributions_parsed(self):
        payload = {
            "model_factor_contributions": [
                {"feature_name": "dti", "shap_contribution": 0.2, "raw_value": 0.35},
            ],
            "model_version": "XGB_V1",
        }
        result = _model_payload_from_decision(payload)
        assert len(result["model_factor_contributions"]) == 1
        assert result["model_factor_contributions"][0]["feature_name"] == "dti"
        assert result["model_version"] == "XGB_V1"

    def test_rows_without_feature_name_skipped(self):
        payload = {
            "model_factor_contributions": [
                {"feature_name": "", "shap_contribution": 0.1},
                {"shap_contribution": 0.2},
            ]
        }
        result = _model_payload_from_decision(payload)
        assert result["model_factor_contributions"] == []

    def test_non_dict_rows_skipped(self):
        payload = {"model_factor_contributions": ["not_a_dict", 42]}
        result = _model_payload_from_decision(payload)
        assert result["model_factor_contributions"] == []

    def test_direction_inferred_from_shap(self):
        payload = {
            "model_factor_contributions": [
                {"feature_name": "dti", "shap_contribution": 0.3, "raw_value": 0.5},
                {"feature_name": "credit_score", "shap_contribution": -0.2, "raw_value": 720},
            ]
        }
        result = _model_payload_from_decision(payload)
        contribs = {c["feature_name"]: c for c in result["model_factor_contributions"]}
        assert contribs["dti"]["direction"] == "increase_default_risk"
        assert contribs["credit_score"]["direction"] == "decrease_default_risk"


# ---------------------------------------------------------------------------
# _model_summary_from_contributions
# ---------------------------------------------------------------------------


class TestModelSummaryFromContributions:
    def test_empty_returns_none(self):
        assert _model_summary_from_contributions([]) is None

    def test_only_increases(self):
        contribs = [
            {"feature_name": "dti", "raw_value": "0.5", "shap_contribution": 0.3, "direction": "increase_default_risk"},
        ]
        result = _model_summary_from_contributions(contribs)
        assert result is not None
        assert "increased predicted default risk" in result

    def test_increases_and_decreases(self):
        contribs = [
            {"feature_name": "dti", "raw_value": "0.5", "shap_contribution": 0.3, "direction": "increase_default_risk"},
            {"feature_name": "credit_score", "raw_value": "720", "shap_contribution": -0.2, "direction": "decrease_default_risk"},
        ]
        result = _model_summary_from_contributions(contribs)
        assert "while" in result
        assert "Model factors" in result

    def test_only_decreases_no_increases_fragment(self):
        contribs = [
            {"feature_name": "credit_score", "raw_value": "780", "shap_contribution": -0.4, "direction": "decrease_default_risk"},
        ]
        result = _model_summary_from_contributions(contribs)
        assert result is not None
        assert "reduced predicted default risk" in result


# ---------------------------------------------------------------------------
# _serialise_model_fragment
# ---------------------------------------------------------------------------


class TestSerialiseModelFragment:
    def test_single_row(self):
        rows = [{"feature_name": "dti", "raw_value": "0.4"}]
        result = _serialise_model_fragment(rows, "increased risk")
        assert result == "dti (0.4) increased risk"

    def test_two_rows_joined_with_and(self):
        rows = [
            {"feature_name": "dti", "raw_value": "0.4"},
            {"feature_name": "income", "raw_value": "50000"},
        ]
        result = _serialise_model_fragment(rows, "increased risk")
        assert "dti (0.4)" in result
        assert "income (50000)" in result
        assert "and" in result


# ---------------------------------------------------------------------------
# _append_model_summary / _append_drift_summary / _append_warning_and_model_summary
# ---------------------------------------------------------------------------


class TestAppendHelpers:
    def test_append_model_summary_with_summary(self):
        model_details = {"model_summary": "Model says high risk."}
        result = _append_model_summary("Application denied.", model_details)
        assert "Model says high risk." in result
        assert "Application denied." in result

    def test_append_model_summary_no_summary(self):
        model_details = {"model_summary": None}
        result = _append_model_summary("Application denied.", model_details)
        assert result == "Application denied."

    def test_append_drift_summary_no_steps(self):
        result = _append_drift_summary("Decision made.", [])
        assert result == "Decision made."

    def test_append_drift_summary_with_drifted_features(self):
        entry = MagicMock()
        entry.output_snapshot = {
            "drifted_features": [
                {"feature_name": "dti"},
                {"feature_name": "income"},
            ]
        }
        result = _append_drift_summary("Decision made.", [entry])
        assert "drift warning" in result
        assert "dti" in result

    def test_append_drift_summary_no_named_features(self):
        entry = MagicMock()
        entry.output_snapshot = {"drifted_features": []}
        result = _append_drift_summary("Decision made.", [entry])
        assert "drift warning" in result

    def test_append_warning_and_model_summary_chained(self):
        model_details = {"model_summary": "High DTI."}
        result = _append_warning_and_model_summary("Denied.", [], model_details)
        assert "High DTI." in result
        assert "Denied." in result


# ---------------------------------------------------------------------------
# _model_details
# ---------------------------------------------------------------------------


class TestModelDetails:
    def test_returns_from_decision_output_when_contributions_present(self):
        decision_output = {
            "model_factor_contributions": [
                {"feature_name": "dti", "shap_contribution": 0.2, "raw_value": 0.4}
            ],
            "model_version": "XGB_V1",
            "model_summary": "DTI high.",
        }
        result = _model_details([], decision_output)
        assert result["model_version"] == "XGB_V1"

    def test_falls_back_to_audit_ml_scoring_entry(self):
        entry = MagicMock()
        entry.step = "ML_SCORING"
        entry.output_snapshot = {
            "model_factor_contributions": [
                {"feature_name": "credit_score", "shap_contribution": -0.3, "raw_value": 720}
            ],
            "model_version": "XGB_V1",
            "model_summary": "Credit score high.",
        }
        result = _model_details([entry], {})
        assert result["model_version"] == "XGB_V1"

    def test_empty_returns_empty_structure(self):
        result = _model_details([], {})
        assert result["model_factor_contributions"] == []
        assert result["model_version"] is None
        assert result["model_summary"] is None
