import os
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from api.auth import (
    OIDCAuth,
    OIDCConfig,
    CompositeAuth,
    RateLimiter,
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    APIKeyAuth,
)
from api.main import app


def test_missing_api_key_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "test-api-key-for-ci:read-write")
    client = TestClient(app)
    response = client.get("/api/v1/status/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 401


def test_invalid_api_key_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "test-api-key-for-ci:read-write")
    client = TestClient(app)
    response = client.get(
        "/api/v1/status/00000000-0000-0000-0000-000000000000",
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_read_only_key_cannot_submit_application(monkeypatch, sample_apply_payload) -> None:
    monkeypatch.setenv("API_KEYS", "test-api-key-for-ci:read")
    client = TestClient(app)
    response = client.post(
        "/api/v1/apply-loan",
        json=sample_apply_payload,
        headers={"X-API-Key": "test-api-key-for-ci"},
    )
    assert response.status_code == 403


def test_health_no_auth_required() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_metrics_no_auth_required() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200


def test_oidc_auth_init() -> None:
    config = OIDCConfig(
        issuer="https://example.com",
        audience="auditlend",
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    auth = OIDCAuth(config=config)
    assert auth._get_config().issuer == "https://example.com"
    assert auth._get_config().audience == "auditlend"


def test_oidc_auth_no_config_returns_503(monkeypatch) -> None:
    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    monkeypatch.delenv("OIDC_JWKS_URL", raising=False)
    auth = OIDCAuth()

    with pytest.raises(Exception) as exc:
        import asyncio
        asyncio.run(auth(None))

    assert exc.value.status_code == 503


def test_oidc_auth_missing_token_returns_401(monkeypatch) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr("api.auth.OAUTH_AVAILABLE", True)

    config = OIDCConfig(
        issuer="https://example.com",
        audience="auditlend",
        jwks_url="https://example.com/.well-known/jwks.json",
    )
    auth = OIDCAuth(config=config)

    with pytest.raises(HTTPException) as exc:
        import asyncio
        asyncio.run(auth(None))

    assert exc.value.status_code == 401


def test_composite_auth_tries_both_methods(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "test-key:read-write")
    composite = CompositeAuth()

    result = None
    async def call():
        nonlocal result
        result = await composite("test-key", None)
    import asyncio
    asyncio.run(call())
    assert result == "test-key"


def test_composite_auth_fails_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "test-key:read-write")
    composite = CompositeAuth()

    with pytest.raises(Exception) as exc:
        import asyncio
        asyncio.run(composite(None, None))

    assert exc.value.status_code == 401


def test_rate_limiter_allows_requests_within_window() -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint(request: Request):
        await limiter(request)
        return {"ok": True}

    client = TestClient(app)
    for _ in range(5):
        response = client.get("/test")
        assert response.status_code == 200


def test_rate_limiter_blocks_after_limit_exceeded() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint(request: Request):
        await limiter(request)
        return {"ok": True}

    client = TestClient(app)
    response1 = client.get("/test")
    assert response1.status_code == 200
    response2 = client.get("/test")
    assert response2.status_code == 200
    response3 = client.get("/test")
    assert response3.status_code == 429


def test_rate_limiter_resets_after_window() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=1)
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint(request: Request):
        await limiter(request)
        return {"ok": True}

    client = TestClient(app)
    response1 = client.get("/test")
    assert response1.status_code == 200
    response2 = client.get("/test")
    assert response2.status_code == 200
    response3 = client.get("/test")
    assert response3.status_code == 429

    time.sleep(1.1)

    response4 = client.get("/test")
    assert response4.status_code == 200


def test_security_headers_middleware_adds_all_required_headers() -> None:
    test_app = FastAPI()

    @test_app.get("/test")
    async def test_endpoint():
        return {"ok": True}

    test_app.add_middleware(SecurityHeadersMiddleware)

    client = TestClient(test_app)
    response = client.get("/test")

    assert response.headers.get("strict-transport-security") == "max-age=31536000; includeSubDomains"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("content-security-policy") == "default-src 'self'"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("permissions-policy") == "geolocation=(), microphone=(), camera=()"


def test_request_size_limit_middleware_blocks_large_body() -> None:
    test_app = FastAPI()

    @test_app.post("/test")
    async def test_endpoint():
        return {"ok": True}

    test_app.add_middleware(RequestSizeLimitMiddleware, max_size_bytes=100)

    client = TestClient(test_app)
    response = client.post("/test", content=b"x" * 200)
    assert response.status_code == 413

    response2 = client.post("/test", content=b"x" * 50)
    assert response2.status_code == 200


def test_vault_client_fallback_to_env_vars() -> None:
    from services.vault import VaultClient

    os.environ["PII_ENCRYPTION_KEY"] = "test-key-hex"
    os.environ["PAN_HASH_SALT"] = "test-salt"

    client = VaultClient(url="", token="")
    assert not client.available

    key = client.get_pii_encryption_key()
    assert key == "test-key-hex"

    salt = client.get_pan_salt()
    assert salt == "test-salt"


def test_vault_client_graceful_degradation(monkeypatch) -> None:
    monkeypatch.delenv("PII_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("PAN_HASH_SALT", raising=False)

    from services.vault import VaultClient

    client = VaultClient(url="http://localhost:8200", token="bad-token")
    key = client.get_pii_encryption_key()
    assert key is None


def test_apikey_auth_continues_working(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "legacy-key:read-write")
    auth = APIKeyAuth()
    result = None
    async def call():
        nonlocal result
        result = await auth("legacy-key")
    import asyncio
    asyncio.run(call())
    assert result == "legacy-key"


def test_oidc_auth_with_python_jose_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("api.auth.OAUTH_AVAILABLE", False)
    auth = OIDCAuth(config=OIDCConfig(
        issuer="https://example.com",
        audience="test",
        jwks_url="https://example.com/jwks",
    ))
    with pytest.raises(Exception) as exc:
        import asyncio
        asyncio.run(auth(None))
    assert exc.value.status_code == 503
