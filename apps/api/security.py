from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import jwt

from packages.shared.schemas.common import TenantContext


@dataclass(frozen=True, slots=True)
class AuthConfig:
    issuer: str | None
    audience: str | None
    alg: str
    secret: str | None


def _auth_config() -> AuthConfig:
    return AuthConfig(
        issuer=os.getenv("JWT_ISSUER"),
        audience=os.getenv("JWT_AUDIENCE"),
        alg=os.getenv("JWT_ALG", "HS256"),
        secret=os.getenv("JWT_SECRET"),
    )


def is_dev_mode() -> bool:
    return os.getenv("APP_ENV", os.getenv("ENV", "dev")).lower() in {"dev", "local"}


def decode_jwt(token: str) -> dict[str, Any]:
    cfg = _auth_config()
    if not cfg.secret:
        raise ValueError("JWT_SECRET no configurado")

    options = {
        "require": ["exp", "iat", "sub"],
        "verify_signature": True,
        "verify_exp": True,
        "verify_iat": True,
        "verify_aud": bool(cfg.audience),
        "verify_iss": bool(cfg.issuer),
    }

    return jwt.decode(
        token,
        cfg.secret,
        algorithms=[cfg.alg],
        audience=cfg.audience,
        issuer=cfg.issuer,
        options=options,
    )


def tenant_context_from_claims(claims: dict[str, Any]) -> TenantContext:
    tenant_id = claims.get("tenant_id") or claims.get("tid")
    user_id = claims.get("sub")
    if not tenant_id or not user_id:
        raise ValueError("JWT missing tenant_id/tid or sub")

    # roles/scopes opcionales
    role = (claims.get("role") or None)
    roles = claims.get("roles")
    if role is None:
        if isinstance(roles, list) and roles:
            role = str(roles[0])
        else:
            role = "user"

    scopes_raw = claims.get("scopes") or []
    if isinstance(scopes_raw, str):
        scopes = [s.strip() for s in scopes_raw.split(" ") if s.strip()]
    elif isinstance(scopes_raw, list):
        scopes = [str(s) for s in scopes_raw if str(s).strip()]
    else:
        scopes = []

    try:
        uid = UUID(str(user_id))
    except Exception as e:
        raise ValueError("JWT sub debe ser UUID") from e

    return TenantContext(
        tenant_id=str(tenant_id),
        user_id=uid,
        role=str(role),
        scopes=scopes,
    )


def require_scope(ctx: TenantContext, scope: str) -> None:
    if scope not in (ctx.scopes or []):
        raise PermissionError(f"Missing scope: {scope}")


def require_role(ctx: TenantContext, role: str) -> None:
    # orden simple
    order = {"user": 1, "admin": 2, "owner": 3}
    if order.get(str(ctx.role), 0) < order.get(role, 0):
        raise PermissionError(f"Missing role: {role}")

