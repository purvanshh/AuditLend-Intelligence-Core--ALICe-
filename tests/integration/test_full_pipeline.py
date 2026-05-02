import asyncio
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

import engine.decision as decision_module
from models.application import LoanApplication
from engine.scoring import MLScoringResult
from services import FailureType, ServiceResult
from tests.conftest import encrypted_application_fields
from worker.tasks import process_application as task_module


class FakeRedis:
    async def aclose(self) -> None:
        return None


def _insert_application(engine, user_data, failure_flags=None) -> str:
    application_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(
            LoanApplication(
                id=application_id,
                idempotency_key=f"pipeline-{application_id}",
                **encrypted_application_fields(user_data),
                failure_flags=failure_flags or {},
                status="PENDING",
            )
        )
        session.commit()
    return str(application_id)


def test_full_worker_pipeline_success(monkeypatch, clean_database, sample_user_data) -> None:
    application_id = _insert_application(clean_database, sample_user_data)

    async def fake_credit_fetch(*args, **kwargs):
        return ServiceResult(success=True, data={"credit_score": 800}, raw_response={"credit_score": 800})

    async def fake_bank_analyze(*args, **kwargs):
        return ServiceResult(success=True, data={"income_stability": 0.9}, raw_response={"income_stability": 0.9})

    async def fake_gst_verify(*args, **kwargs):
        return ServiceResult(success=True, data={"gst_compliant": True}, raw_response={"gst_compliant": True})

    monkeypatch.setattr(task_module.redis_async, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(task_module.CreditBureauService, "fetch", fake_credit_fetch)
    monkeypatch.setattr(task_module.BankAnalyzerService, "analyze", fake_bank_analyze)
    monkeypatch.setattr(task_module.GstVerifierService, "verify", fake_gst_verify)

    result = asyncio.run(task_module._process_application(application_id))

    assert result["status"] == "COMPLETED"
    assert result["decision"] == "APPROVE"
    with clean_database.connect() as connection:
        row = connection.execute(
            text("SELECT status, decision, confidence FROM loan_applications WHERE id = :id"),
            {"id": application_id},
        ).one()
        external_count = connection.scalar(text("SELECT count(*) FROM external_data"))
        audit_count = connection.scalar(text("SELECT count(*) FROM audit_logs"))

    assert row.status == "COMPLETED"
    assert row.decision == "APPROVE"
    assert float(row.confidence) == 1.0
    assert external_count == 3
    assert audit_count >= 5


def test_full_worker_pipeline_all_failures_routes_manual_review(monkeypatch, clean_database, sample_user_data) -> None:
    application_id = _insert_application(
        clean_database,
        sample_user_data,
        {
            "credit_bureau": "TIMEOUT",
            "bank_analyzer": "FORMAT_ERROR",
            "gst_verifier": "PAN_MISMATCH",
        },
    )

    async def fake_fetch_external_data(app_id, user_data, failure_flags, redis_client):
        return (
            ServiceResult(
                success=False,
                data={"credit_score": 600},
                failure_type=FailureType.TIMEOUT,
                raw_response={"error": "timeout"},
                fallback_used=True,
            ),
            ServiceResult(
                success=False,
                data=None,
                failure_type=FailureType.FORMAT_ERROR,
                raw_response={"error": "bad format"},
                fallback_used=True,
            ),
            ServiceResult(
                success=False,
                data={"gst_compliant": False},
                failure_type=FailureType.PAN_MISMATCH,
                raw_response={"match": False},
                fallback_used=True,
            ),
        )

    monkeypatch.setattr(task_module.redis_async, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(task_module, "_fetch_external_data", fake_fetch_external_data)

    result = asyncio.run(task_module._process_application(application_id))

    assert result["status"] == "MANUAL_REVIEW"
    assert result["decision"] == "NEEDS_REVIEW"
    assert result["confidence"] < 0.6


def test_full_worker_pipeline_ml_path_writes_audit_entry(monkeypatch, clean_database, sample_user_data) -> None:
    application_id = _insert_application(clean_database, sample_user_data)

    async def fake_credit_fetch(*args, **kwargs):
        return ServiceResult(success=True, data={"credit_score": 790}, raw_response={"credit_score": 790})

    async def fake_bank_analyze(*args, **kwargs):
        return ServiceResult(
            success=True,
            data={"income_stability": 0.92, "monthly_inflow": 120000, "monthly_outflow": 45000},
            raw_response={"income_stability": 0.92, "monthly_inflow": 120000, "monthly_outflow": 45000},
        )

    async def fake_gst_verify(*args, **kwargs):
        return ServiceResult(success=True, data={"gst_compliant": True}, raw_response={"gst_compliant": True})

    class FakeMLScorer:
        def score(self, *args, **kwargs):
            return MLScoringResult(
                attempted=True,
                used=True,
                fallback_used=False,
                fallback_reason=None,
                error_type=None,
                risk_score=81.6,
                predicted_default_probability=0.22,
                calibrated_default_probability=0.184,
                model_confidence=0.816,
                model_version="XGB_V1",
                selected_candidate="lightgbm",
                score_breakdown=[
                    "ml_default_probability (raw) = 0.2200",
                    "ml_default_probability (calibrated) = 0.1840",
                    "risk_score (ml_mapped) = 81.60",
                ],
                model_factor_contributions=[
                    {
                        "feature_name": "Credit Score",
                        "raw_value": "790",
                        "shap_contribution": -0.19,
                        "direction": "decrease_default_risk",
                    }
                ],
                model_summary="Model factors: Credit Score (790) reduced predicted default risk.",
                drift_report={
                    "status": "WARNING",
                    "model_version": "XGB_V1",
                    "alert_count": 1,
                    "drifted_features": [
                        {
                            "feature_name": "loan_amount_to_income",
                            "p_value": 0.001,
                        }
                    ],
                },
            )

    monkeypatch.setenv("ML_ENABLED", "true")
    monkeypatch.setattr(task_module.redis_async, "from_url", lambda *args, **kwargs: FakeRedis())
    monkeypatch.setattr(task_module.CreditBureauService, "fetch", fake_credit_fetch)
    monkeypatch.setattr(task_module.BankAnalyzerService, "analyze", fake_bank_analyze)
    monkeypatch.setattr(task_module.GstVerifierService, "verify", fake_gst_verify)
    monkeypatch.setattr(decision_module, "get_ml_scorer_from_env", lambda: FakeMLScorer())

    result = asyncio.run(task_module._process_application(application_id))

    assert result["status"] == "COMPLETED"
    assert result["decision"] == "APPROVE"
    assert result["rule_version"] == "RULE_SET_V2"
    assert result["model_version"] == "XGB_V1"
    assert result["scoring_strategy"] == "ml"

    with clean_database.connect() as connection:
        steps = connection.execute(
            text("SELECT step FROM audit_logs WHERE application_id = :id ORDER BY id"),
            {"id": application_id},
        ).scalars().all()
        ml_snapshot = connection.execute(
            text("SELECT output_snapshot FROM audit_logs WHERE application_id = :id AND step = 'ML_SCORING'"),
            {"id": application_id},
        ).scalar_one()
        drift_snapshot = connection.execute(
            text("SELECT output_snapshot FROM audit_logs WHERE application_id = :id AND step = 'DRIFT_DETECTED'"),
            {"id": application_id},
        ).scalar_one()

    assert "ML_SCORING" in steps
    assert "DRIFT_DETECTED" in steps
    assert ml_snapshot["model_version"] == "XGB_V1"
    assert ml_snapshot["fallback_used"] is False
    assert drift_snapshot["status"] == "WARNING"
