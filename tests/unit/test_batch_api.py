from __future__ import annotations

import os

os.environ.setdefault("PII_ENCRYPTION_KEY", "02468ace02468ace02468ace02468ace02468ace02468ace02468ace02468ace")
os.environ.setdefault("PAN_HASH_SALT", "test-salt-for-ci")
os.environ.setdefault("API_KEYS", "test-api-key-for-ci:read-write")

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_ml_available():
    import api.routes.batch as batch_module
    batch_module._ML_AVAILABLE = None
    yield
    batch_module._ML_AVAILABLE = None


def _make_auth_header() -> dict[str, str]:
    return {"X-API-Key": "test-api-key-for-ci"}


def test_batch_predict_with_sample_data(monkeypatch) -> None:
    import api.routes.batch as batch_module
    monkeypatch.setattr("api.routes.batch._check_ml_available", lambda: True)
    batch_module._ML_AVAILABLE = True

    payload = {
        "features": [
            {"loan_amount": 10000, "credit_score": 720},
            {"loan_amount": 50000, "credit_score": 620},
        ],
        "model_version": "XGB_V1",
    }
    response = client.post("/api/v1/batch/predict", json=payload, headers=_make_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
    assert data["latency_ms"] >= 0


def test_batch_predict_empty_features(monkeypatch) -> None:
    import api.routes.batch as batch_module
    monkeypatch.setattr("api.routes.batch._check_ml_available", lambda: True)
    batch_module._ML_AVAILABLE = True

    payload = {"features": [], "model_version": "XGB_V1"}
    response = client.post("/api/v1/batch/predict", json=payload, headers=_make_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["batch_size"] == 0
    assert data["latency_ms"] == 0.0


def test_batch_predict_without_ml(monkeypatch) -> None:
    import api.routes.batch as batch_module
    monkeypatch.setattr("api.routes.batch._check_ml_available", lambda: False)
    batch_module._ML_AVAILABLE = False

    payload = {
        "features": [{"loan_amount": 10000}],
        "model_version": "XGB_V1",
    }
    response = client.post("/api/v1/batch/predict", json=payload, headers=_make_auth_header())
    assert response.status_code == 424
    data = response.json()
    assert "detail" in data
    assert "ML model not available" in data["detail"]


def test_batch_status_endpoint(monkeypatch) -> None:
    import api.routes.batch as batch_module
    monkeypatch.setattr("api.routes.batch._check_ml_available", lambda: True)
    batch_module._ML_AVAILABLE = True

    response = client.get("/api/v1/batch/status", headers=_make_auth_header())
    assert response.status_code == 200
    data = response.json()
    assert "ml_available" in data
    assert "cache_stats" in data
