from __future__ import annotations

from uuid import UUID

import asyncpg

from packages.shared.schemas.common import TenantContext


class IngestionService:
    """Gestiona estados del pipeline de ingesta: pending → processing → ready/failed.

    Usa RLS por tenant: requiere setear app.tenant_id por conexión.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def start_processing(self, *, ctx: TenantContext, doc_id: UUID) -> None:
        """Transición: pending → processing (o permanece en processing si ya está)."""
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id)
            await conn.execute(
                """
                UPDATE documents
                SET status = 'processing'
                WHERE tenant_id = $1 AND doc_id = $2::uuid AND status IN ('pending','processing')
                """,
                ctx.tenant_id,
                str(doc_id),
            )

    async def mark_ready(self, *, ctx: TenantContext, doc_id: UUID) -> None:
        """Transición: processing → ready."""
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id)
            await conn.execute(
                """
                UPDATE documents
                SET status = 'ready'
                WHERE tenant_id = $1 AND doc_id = $2::uuid
                """,
                ctx.tenant_id,
                str(doc_id),
            )

    async def mark_failed(self, *, ctx: TenantContext, doc_id: UUID) -> None:
        """Transición: cualquier estado → failed."""
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id)
            await conn.execute(
                """
                UPDATE documents
                SET status = 'failed'
                WHERE tenant_id = $1 AND doc_id = $2::uuid
                """,
                ctx.tenant_id,
                str(doc_id),
            )

