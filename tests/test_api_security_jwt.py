import os
from uuid import uuid4

import jwt
import pytest
from starlette.requests import Request

from apps.api.deps import get_ctx


def _make_token(*, tenant_id: str, user_id: str, scopes=None, role="user") -> str:
    secret = os.getenv("JWT_SECRET", "test-secret")
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "scopes": scopes or ["chat:ask"],
        "iss": "test",
        "aud": "test",
        "iat": 1700000000,
        "exp": 9999999999,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "query_string": b"",
        "client": ("test", 123),
        "server": ("test", 80),
        "scheme": "http",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_missing_auth_is_401(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("JWT_ISSUER", "test")
    monkeypatch.setenv("JWT_AUDIENCE", "test")

    with pytest.raises(Exception) as e:
        await get_ctx(_fake_request(), authorization=None)  # type: ignore[arg-type]

    # FastAPI lanza HTTPException; no importamos el tipo para mantener test simple
    assert "Missing Authorization" in str(e.value)


@pytest.mark.asyncio
async def test_jwt_builds_ctx_and_ignores_x_tenant_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("JWT_ISSUER", "test")
    monkeypatch.setenv("JWT_AUDIENCE", "test")

    tenant_id = str(uuid4())
    user_id = str(uuid4())
    token = _make_token(tenant_id=tenant_id, user_id=user_id)

    ctx = await get_ctx(
        _fake_request(),
        authorization=f"Bearer {token}",
        x_tenant_id=str(uuid4()),
        x_user_id=str(uuid4()),
    )

    assert str(ctx.tenant_id) == tenant_id
    assert str(ctx.user_id) == user_id
