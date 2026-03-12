from __future__ import annotations

import os

import asyncpg
from fastapi import APIRouter, Depends

from apps.api.deps import get_db_pool

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(pool: asyncpg.Pool = Depends(get_db_pool)) -> dict[str, str]:
    """Readiness check (production-grade).

    Valida:
    - DB responde
    - row_security = on
    - la sesi5n puede setear app.tenant_id (convenci9n RLS) y leer current_setting
    - query m8nima sobre una tabla protegida (runs) para comprobar permisos/esquema

    Nota: no valida aislamiento cross-tenant en runtime; eso se cubre con tests de integraci9n.
    """

    tenant = os.getenv("READINESS_TENANT_ID", "00000000-0000-0000-0000-000000000000")

    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

            row_security = await conn.fetchval("SHOW row_security")
            if str(row_security).lower() != "on":
                return {"status": "not_ready", "reason": "row_security_off"}

            # set tenant context (defense: set_config debe funcionar por conexi9n)
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant))
            current = await conn.fetchval("SELECT current_setting('app.tenant_id', true)")
            if str(current or "") != str(tenant):
                return {"status": "not_ready", "reason": "tenant_context_not_set"}

            # query m8nima (0 filas ok)
            await conn.fetchval("SELECT 1 FROM runs LIMIT 1")

    except Exception:
        # No revelar detalles sensibles en readiness
        return {"status": "not_ready"}

    return {"status": "ready"}
