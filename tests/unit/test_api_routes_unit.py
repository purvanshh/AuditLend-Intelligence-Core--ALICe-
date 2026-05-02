from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, Response

from api.routes import applications, decisions, explanations
from api.schemas.application import ApplyLoanRequest


class FakeAsyncSession:
    def __init__(self, *, existing=None, inserted_key: str | None = "ok") -> None:
        self.existing = existing
        self.inserted_key = inserted_key
        self.added: list[object] = []
        self.flushed = False
        self.committed = False
        self.rolled_back = False

    async def get(self, model, key):
        return self.existing

    def add(self, obj) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushed = True

    async def scalar(self, statement):
        return self.inserted_key

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def scalars(self, statement):
        return SimpleNamespace(all=lambda: self.existing or [])


class FakePiiService:
    def encrypt(self, user_data):
        return (b"cipher", b"nonce")

    def hash_pan(self, pan: str) -> str:
        return "h" * 64


def _request() -> ApplyLoanRequest:
    return ApplyLoanRequest.model_validate(
        {
            "idempotency_key": "idem-1",
            "user_data": {
                "name": "Jane Doe",
                "pan": "ABCDE1234F",
                "monthly_income": 100000,
                "existing_emis": 20000,
                "loan_amount": 300000,
                "tenure_months": 36,
            },
        }
    )


@pytest.mark.asyncio
async def test_apply_loan_creates_application_and_outbox(monkeypatch) -> None:
    session = FakeAsyncSession()
    async def fake_get(key):
        return None

    async def fake_set(key, payload):
        return None

    monkeypatch.setattr(applications, "_redis_idempotency_get", fake_get)
    monkeypatch.setattr(applications, "_redis_idempotency_set", fake_set)
    monkeypatch.setattr(applications, "pii_service_from_env", lambda: FakePiiService())

    response = Response()
    payload = _request()
    result = await applications.apply_loan(payload, response, session)

    assert result.status == "PENDING"
    assert session.flushed is True
    assert session.committed is True
    assert len(session.added) == 2


@pytest.mark.asyncio
async def test_apply_loan_returns_cached_idempotent_response(monkeypatch) -> None:
    cached = {
        "_request_hash": applications._payload_hash(_request(), "idem-1"),
        "public": {
            "application_id": "abc",
            "status": "PENDING",
            "message": "Application received and queued for processing",
        },
    }
    async def fake_get(key):
        return cached

    monkeypatch.setattr(applications, "_redis_idempotency_get", fake_get)
    response = Response()

    result = await applications.apply_loan(_request(), response, FakeAsyncSession())

    assert response.status_code == 200
    assert result.application_id == "abc"


@pytest.mark.asyncio
async def test_apply_loan_rejects_payload_change_for_existing_record(monkeypatch) -> None:
    existing = SimpleNamespace(
        response={
            "_request_hash": "other-hash",
            "public": {"application_id": "abc", "status": "PENDING", "message": "queued"},
        }
    )
    async def fake_get(key):
        return None

    monkeypatch.setattr(applications, "_redis_idempotency_get", fake_get)

    with pytest.raises(HTTPException) as exc:
        await applications.apply_loan(_request(), Response(), FakeAsyncSession(existing=existing))

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_decision_and_explanation_routes_cover_pending_and_complete_states(monkeypatch) -> None:
    app_id = uuid4()
    pending_session = FakeAsyncSession(existing=SimpleNamespace(id=app_id, status="PENDING", decision=None))
    decision_response = await decisions.get_decision(str(app_id), pending_session)
    explanation_response = await explanations.get_explanation(str(app_id), pending_session)
    assert decision_response.status_code == 202
    assert explanation_response.status_code == 202

    complete_application = SimpleNamespace(id=app_id, status="COMPLETED", decision="APPROVE", confidence=0.91)
    audit_entry = SimpleNamespace(
        step="DECISION_CALCULATION",
        output_snapshot={
            "data_reliability": 1.0,
            "risk_score": 85.0,
            "factors": ["risk_score (computed) = 85.00"],
            "rule_version": "RULE_SET_V1",
            "model_version": "XGB_V1",
            "scoring_strategy": "ml",
            "ab_test_arm": "ml",
        },
    )
    complete_session = FakeAsyncSession(existing=[audit_entry])

    async def fake_get(model, key):
        return complete_application

    complete_session.get = fake_get  # type: ignore[method-assign]
    monkeypatch.setattr(
        explanations,
        "build_explanation",
        lambda audit_entries, decision_output: {
            "decision": "APPROVE",
            "summary": "ok",
            "factors": [],
            "timeline": [],
            "model_factor_contributions": [],
            "rule_version": "RULE_SET_V1",
            "model_version": "XGB_V1",
            "generated_at": datetime.utcnow(),
        },
    )

    async def fake_decision_output(session, app_uuid):
        return audit_entry.output_snapshot

    monkeypatch.setattr(decisions, "_decision_output", fake_decision_output)

    decision_result = await decisions.get_decision(str(app_id), complete_session)
    explanation_result = await explanations.get_explanation(str(app_id), complete_session)

    assert decision_result.decision == "APPROVE"
    assert decision_result.model_version == "XGB_V1"
    assert explanation_result.summary == "ok"
