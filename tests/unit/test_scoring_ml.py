from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from engine import scoring
from services import ServiceResult


@dataclass
class FakeExplanation:
    predicted_default_probability: float
    calibrated_default_probability: float | None
    model_summary: str
    contributions: list[dict]

    def to_audit_payload(self) -> dict:
        return {"model_factor_contributions": list(self.contributions)}


def _service_result(data: dict | None = None) -> ServiceResult:
    return ServiceResult(success=True, data=data or {}, raw_response=data or {})


def test_ml_scorer_success_uses_calibrated_probability(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        scoring,
        "load_manifest",
        lambda path: {"run_id": "XGB_V1", "selected_candidate": "xgboost"},
    )
    monkeypatch.setattr(
        scoring,
        "explain_feature_row",
        lambda *args, **kwargs: FakeExplanation(
            predicted_default_probability=0.22,
            calibrated_default_probability=0.18,
            model_summary="Low utilization reduced default risk.",
            contributions=[{"feature_name": "Credit Score", "shap_contribution": -0.2}],
        ),
    )
    monkeypatch.setattr(
        scoring,
        "_detect_feature_drift",
        lambda *args, **kwargs: {
            "alert_count": 1,
            "drifted_features": [{"feature_name": "loan_amount_to_income"}],
        },
    )

    scorer = scoring.MLScorer(tmp_path / "manifest.json", model_version="XGB_V1")
    result = scorer.score(
        _service_result({"credit_score": 790}),
        _service_result({"income_stability": 0.9}),
        _service_result({"gst_compliant": True}),
        {
            "pan": "ABCDE1234F",
            "monthly_income": 120000,
            "existing_emis": 12000,
            "loan_amount": 300000,
            "tenure_months": 36,
        },
        confidence_threshold=0.6,
    )

    assert result.used is True
    assert result.fallback_used is False
    assert result.risk_score == 82.0
    assert result.calibrated_default_probability == 0.18
    assert result.model_confidence == 0.82
    assert result.model_version == "XGB_V1"
    assert result.model_factor_contributions == [{"feature_name": "Credit Score", "shap_contribution": -0.2}]
    assert result.drift_report is not None
    assert result.drift_report["alert_count"] == 1
    assert "ml_drift_alert (warning) = loan_amount_to_income" in result.score_breakdown


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason", "expected_error_type"),
    [
        ("TIMEOUT", "TIMEOUT", "TIMEOUT"),
        ("FORCE_CONFIDENCE_0.4", "FORCE_LOW_CONFIDENCE", None),
        ("FORCE_LOW_CONFIDENCE", "FORCE_LOW_CONFIDENCE", None),
    ],
)
def test_ml_scorer_deterministic_failure_modes(monkeypatch, tmp_path: Path, failure_mode: str, expected_reason: str, expected_error_type: str | None) -> None:
    monkeypatch.setattr(
        scoring,
        "load_manifest",
        lambda path: {"run_id": "XGB_V1", "selected_candidate": "xgboost"},
    )
    scorer = scoring.MLScorer(tmp_path / "manifest.json", model_version="XGB_V1")

    result = scorer.score(
        _service_result(),
        _service_result(),
        _service_result(),
        {
            "pan": "ABCDE1234F",
            "monthly_income": 120000,
            "existing_emis": 12000,
            "loan_amount": 300000,
            "tenure_months": 36,
        },
        confidence_threshold=0.6,
        failure_mode=failure_mode,
    )

    assert result.used is False
    assert result.fallback_used is True
    assert result.fallback_reason == expected_reason
    assert result.error_type == expected_error_type


def test_build_ml_feature_row_maps_auditlend_inputs() -> None:
    feature_row = scoring._build_ml_feature_row(
        _service_result({"credit_score": 725}),
        _service_result({"income_stability": 0.8, "monthly_inflow": 95000, "monthly_outflow": 30000, "average_balance": 42000}),
        _service_result({"gst_compliant": False}),
        {
            "pan_hash": "hash-1",
            "monthly_income": 100000,
            "existing_emis": 25000,
            "loan_amount": 240000,
            "tenure_months": 24,
            "purpose": "credit_card",
            "home_ownership": "MORTGAGE",
        },
    )

    assert feature_row["grade"] == "B"
    assert feature_row["sub_grade"] == "B2"
    assert feature_row["verification_status"] == "Verified"
    assert feature_row["tax_lien_flag"] == 1.0
    assert feature_row["loan_amount_to_income"] == 2.4
    assert feature_row["installment_to_income"] == 0.1


def test_get_ml_scorer_from_env_uses_manifest_override(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ML_MODEL_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("ML_MODEL_VERSION", "XGB_V1")
    monkeypatch.setattr(
        scoring,
        "load_manifest",
        lambda path: {"run_id": "fallback-run", "selected_candidate": "xgboost"},
    )
    scoring.clear_ml_scorer_cache()

    scorer_instance = scoring.get_ml_scorer_from_env()

    assert scorer_instance is not None
    assert scorer_instance.model_version == "XGB_V1"
    assert scorer_instance.selected_candidate == "xgboost"

    scoring.clear_ml_scorer_cache()
    monkeypatch.delenv("ML_MODEL_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("ML_MODEL_VERSION", raising=False)


def test_preload_ml_scorer_from_env_freezes_missing_model(monkeypatch, tmp_path: Path) -> None:
    official_manifest = tmp_path / "manifest.yaml"
    official_manifest.write_text('{"model_version":"XGB_V1","model_artifact_path":"missing.pkl"}\n', encoding="utf-8")
    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setattr(scoring, "OFFICIAL_MANIFEST_PATH", official_manifest)
    scoring.clear_ml_scorer_cache()

    preloaded = scoring.preload_ml_scorer_from_env()
    loaded = scoring.get_ml_scorer_from_env()

    assert preloaded is None
    assert loaded is None

    scoring.clear_ml_scorer_cache()
    monkeypatch.delenv("ML_ENABLED", raising=False)


def test_scoring_helpers_cover_edge_cases() -> None:
    assert scoring._env_truthy(" yes ") is True
    assert scoring._env_truthy("0") is False
    assert scoring._extract_credit_score_value(_service_result({})) == scoring.DEFAULT_CREDIT_SCORE
    assert scoring._verification_status({"income_stability": 0.8}) == "Verified"
    assert scoring._verification_status({"monthly_inflow": 1000}) == "Source Verified"
    assert scoring._verification_status({}) == "Not Verified"
    assert scoring._grade_from_credit_score(790) == ("A", "A1")
    assert scoring._grade_from_credit_score(500) == ("E", "E5")
    assert scoring._safe_ratio(10.0, 0.0) == 0.0
