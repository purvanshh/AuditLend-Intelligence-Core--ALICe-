import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.responses import Response

from services.metrics import auth_attempts_total, rate_limit_exceeded_total


try:
    from jose import jwt, JWTError
    from jose.constants import Algorithms

    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    jwt = None
    JWTError = Exception
    Algorithms = None


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
logger = structlog.get_logger()


@dataclass
class OIDCConfig:
    issuer: str = ""
    audience: str = ""
    jwks_url: str = ""
    algorithms: list[str] = field(default_factory=lambda: ["RS256"])


class OIDCAuth:
    def __init__(
        self,
        config: OIDCConfig | None = None,
        required_scopes: list[str] | None = None,
    ) -> None:
        self.required_scopes = required_scopes or ["read-write"]
        self._config = config
        self._jwks_cache: dict[str, Any] = {}
        self._jwks_cache_time = 0.0
        self._jwks_cache_ttl = 300.0

    def _load_config_from_env(self) -> OIDCConfig:
        return OIDCConfig(
            issuer=os.environ.get("OIDC_ISSUER", ""),
            audience=os.environ.get("OIDC_AUDIENCE", ""),
            jwks_url=os.environ.get("OIDC_JWKS_URL", ""),
        )

    def _get_config(self) -> OIDCConfig:
        if self._config is not None:
            return self._config
        return self._load_config_from_env()

    async def __call__(
        self,
        authorization: str | None = Security(HTTPBearer(auto_error=False)),
    ) -> dict[str, Any]:
        if not OAUTH_AVAILABLE:
            auth_attempts_total.labels(method="oidc", result="unavailable").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OAuth2/OIDC authentication is not available",
            )

        config = self._get_config()
        if not config.issuer or not config.jwks_url:
            auth_attempts_total.labels(method="oidc", result="misconfigured").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OIDC is not configured",
            )

        if authorization is None:
            auth_attempts_total.labels(method="oidc", result="missing_token").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = authorization.credentials
        claims = await self._validate_token(token)

        auth_attempts_total.labels(method="oidc", result="success").inc()
        return claims

    def _fetch_jwks(self) -> dict[str, Any]:
        now = time.time()
        if self._jwks_cache and (now - self._jwks_cache_time) < self._jwks_cache_ttl:
            return self._jwks_cache

        import httpx

        config = self._get_config()
        response = httpx.get(config.jwks_url, timeout=10.0)
        response.raise_for_status()
        jwks_data = response.json()
        self._jwks_cache = jwks_data
        self._jwks_cache_time = now
        return jwks_data

    async def _validate_token(self, token: str) -> dict[str, Any]:
        config = self._get_config()
        jwks = self._fetch_jwks()
        try:
            claims = jwt.decode(
                token,
                jwks,
                algorithms=config.algorithms,
                issuer=config.issuer,
                audience=config.audience,
            )
        except JWTError as exc:
            auth_attempts_total.labels(method="oidc", result="invalid_token").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {exc}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        if "exp" in claims and claims["exp"] < time.time():
            auth_attempts_total.labels(method="oidc", result="expired").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return claims


class CompositeAuth:
    def __init__(self, required_scopes: list[str] | None = None) -> None:
        self.required_scopes = required_scopes or ["read-write"]
        self._api_key_auth = APIKeyAuth(self.required_scopes)
        self._oidc_auth = OIDCAuth(required_scopes=self.required_scopes)

    async def __call__(
        self,
        api_key: str | None = Security(API_KEY_HEADER),
        authorization: str | None = Security(HTTPBearer(auto_error=False)),
    ) -> str | dict[str, Any]:
        errors: list[str] = []

        if api_key:
            try:
                result = await self._api_key_auth(api_key)
                return result
            except HTTPException as exc:
                errors.append(str(exc.detail))

        if authorization:
            try:
                result = await self._oidc_auth(authorization)
                return result
            except HTTPException as exc:
                errors.append(str(exc.detail))

        if api_key or authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {'; '.join(errors)}",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "ApiKey Bearer"},
        )


class RateLimiter:
    def __init__(self, max_requests: int | None = None, window_seconds: int | None = None) -> None:
        self.max_requests = max_requests or int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "100"))
        self.window_seconds = window_seconds or int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def __call__(self, request: Request) -> None:
        client_key = self._resolve_key(request)
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            timestamps = self._buckets.get(client_key, [])
            timestamps = [t for t in timestamps if t > window_start]

            if len(timestamps) >= self.max_requests:
                rate_limit_exceeded_total.labels(client_key=client_key).inc()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Try again later.",
                )

            timestamps.append(now)
            self._buckets[client_key] = timestamps

    def _resolve_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"


class SecurityHeadersMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                security_headers = [
                    (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"content-security-policy", b"default-src 'self'"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
                ]
                existing_names = {h[0].lower() for h in headers}
                for name, value in security_headers:
                    if name not in existing_names:
                        headers.append((name, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_size_bytes: int | None = None) -> None:
        self.app = app
        self.max_size_bytes = max_size_bytes or int(
            os.environ.get("MAX_REQUEST_SIZE_BYTES", str(1_048_576))
        )

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = 0
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"content-length":
                content_length = int(header_value)
                break

        if content_length > self.max_size_bytes:
            response = Response(
                content='{"detail":"Request body too large"}',
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        original_receive = receive

        async def sized_receive() -> dict[str, Any]:
            message = await original_receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if len(body) > self.max_size_bytes:
                    response = Response(
                        content='{"detail":"Request body too large"}',
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        media_type="application/json",
                    )
                    await response(scope, receive, send)
            return message

        await self.app(scope, sized_receive, send)


class APIKeyAuth:
    def __init__(self, required_scopes: list[str] | None = None) -> None:
        self.required_scopes = required_scopes or ["read-write"]

    async def __call__(self, api_key: str | None = Security(API_KEY_HEADER)) -> str:
        if not api_key:
            auth_attempts_total.labels(method="apikey", result="missing").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        key_scopes = _load_keys().get(api_key)
        if key_scopes is None:
            auth_attempts_total.labels(method="apikey", result="invalid").inc()
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

        if not _has_required_scope(key_scopes, self.required_scopes):
            auth_attempts_total.labels(method="apikey", result="insufficient_scope").inc()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient API key scope"
            )

        auth_attempts_total.labels(method="apikey", result="success").inc()
        return api_key


def _load_keys() -> dict[str, set[str]]:
    keys_str = os.environ.get("API_KEYS", "")
    keys: dict[str, set[str]] = {}
    for raw_entry in keys_str.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        key, scopes = _parse_key_entry(entry)
        keys[key] = scopes
    return keys


def _parse_key_entry(entry: str) -> tuple[str, set[str]]:
    if ":" in entry:
        key, raw_scopes = entry.split(":", 1)
        scopes = {scope.strip() for scope in raw_scopes.split("+") if scope.strip()}
    elif entry.endswith("read-only"):
        key = entry
        scopes = {"read"}
    else:
        key = entry
        scopes = {"read", "write", "read-write"}
    if "read-write" in scopes:
        scopes.update({"read", "write"})
    return key.strip(), scopes


def _has_required_scope(key_scopes: set[str], required_scopes: list[str]) -> bool:
    return all(scope in key_scopes for scope in required_scopes)


require_auth = APIKeyAuth(["write"])
require_read = APIKeyAuth(["read"])
rate_limiter = RateLimiter()
