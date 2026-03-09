from __future__ import annotations

import os

import asyncpg
from fastapi import APIRouter, Depends

from apps.api.deps import get_db_pool

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(pool: asyncpg.Pool = Depends(get_db_pool)) -> dict[str, str]:
    """Readiness check.

    Valida:
    - DB responde
    - row_security=on
    - se puede setear app.tenant_id y consultar una tabla con RLS sin error
    """
    tenant = os.getenv("READINESS_TENANT_ID", "00000000-0000-0000-0000-000000000000")

    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
        row_security = await conn.fetchval("SHOW row_security")
        if str(row_security).lower() != "on":
            return {"status": "not_ready", "reason": "row_security_off"}

        # set tenant y consulta mínima (no importa 0 filas)
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant))
        await conn.fetchval("SELECT 1 FROM runs LIMIT 1")

    return {"status": "ready"}
