from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from engine.decision import DecisionOutput
from engine.rules import Decision
from services import FailureType
from worker.tasks import process_application as task_module


def test_worker_helper_functions_cover_fallback_and_redaction() -> None:
    assert task_module._status_for_decision(
        DecisionOutput(
            decision=Decision.NEEDS_REVIEW,
            confidence=0.4,
            data_reliability=0.7,
            risk_score=55.0,
            factors=[],
            penalty_reasons=[],
            rule_version="RULE_SET_V1",
            requires_manual_review=True,
        )
    ) == "MANUAL_REVIEW"
    assert task_module._failure_flag({"credit_bureau": "TIMEOUT"}, "credit_bureau") == FailureType.TIMEOUT
    assert task_module._fallback_used_for_reconstructed_result(FailureType.NO_RECORD) is True
    assert task_module._fallback_used_for_reconstructed_result(None) is False

    redacted = task_module._redact_user_data(
        {
            "pan": "ABCDE1234F",
            "name": "Jane",
            "monthly_income": 100000,
            "nested": {"loan_amount": 400000},
            "list": [{"existing_emis": 1000}],
        }
    )
    assert redacted["pan"] == "***REDACTED***"
    assert redacted["nested"]["loan_amount"] == "***REDACTED***"
    assert redacted["list"][0]["existing_emis"] == "***REDACTED***"


def test_mark_manual_review_after_system_error_updates_existing_application(monkeypatch) -> None:
    application = SimpleNamespace(id=uuid4(), status="PENDING", decision=None)
    fake_session = SimpleNamespace(get=lambda model, key: application)
    audit_calls: list[dict] = []

    @contextmanager
    def fake_get_sync_session():
        yield fake_session

    monkeypatch.setattr(task_module, "get_sync_session", fake_get_sync_session)
    monkeypatch.setattr(task_module.loan_applications_total, "labels", lambda **kwargs: SimpleNamespace(inc=lambda: None))
    monkeypatch.setattr(task_module, "write_audit_entry", lambda **kwargs: audit_calls.append(kwargs))

    task_module._mark_manual_review_after_system_error(str(application.id), RuntimeError("boom"), "SYSTEM_ERROR")

    assert application.status == "MANUAL_REVIEW"
    assert application.decision == "NEEDS_REVIEW"
    assert audit_calls[0]["step"] == "SYSTEM_ERROR"
