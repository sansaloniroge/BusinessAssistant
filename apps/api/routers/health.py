from __future__ import annotations

from fastapi import APIRouter, Depends
import asyncpg

from apps.api.deps import get_db_pool

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(pool: asyncpg.Pool = Depends(get_db_pool)) -> dict[str, str]:
    """Readiness check mínima.

    V1: comprueba que la DB responde y que la sesión tiene row_security=on.
    (Más adelante: vector_store health, cola, etc.)
    """
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
        row_security = await conn.fetchval("SHOW row_security")

    if str(row_security).lower() != "on":
        return {"status": "not_ready", "reason": "row_security_off"}

    return {"status": "ready"}
