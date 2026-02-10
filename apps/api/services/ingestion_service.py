from __future__ import annotations

from uuid import UUID, uuid5

import asyncpg

from packages.shared.schemas.common import TenantContext


class IngestionService:
    """Gestiona estados del pipeline de ingesta: pending → processing → ready/failed.

    Usa RLS por tenant: requiere setear app.tenant_id por conexión.

    Idempotencia:
    - chunk_id estable por doc_id + índice (uuid5 sobre namespace fijo).
    - tracking de intentos de ingesta en documents.metadata.
    """

    # Namespace fijo para generar uuid5 deterministas de chunks
    CHUNK_UUID_NAMESPACE = UUID("00000000-0000-0000-0000-000000000000")

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

    @classmethod
    def stable_chunk_id(cls, *, doc_id: UUID, index: int) -> str:
        """Genera un chunk_id estable a partir de doc_id + índice.

        Usa UUIDv5 sobre un namespace fijo. Determinista: mismo doc_id+index → mismo chunk_id.
        """
        if index < 0:
            raise ValueError("index debe ser >= 0")
        name = f"{doc_id}:{index}"
        return str(uuid5(cls.CHUNK_UUID_NAMESPACE, name))

    async def start_attempt(self, *, ctx: TenantContext, doc_id: UUID, attempt_id: UUID) -> None:
        """Registra inicio de intento de ingesta en documents.metadata.

        Estructura en metadata.ingestion:
        { attempts: [attempt_id...], last_attempt_id, last_attempt_at }
        """
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id)
            await conn.execute(
                """
                UPDATE documents
                SET metadata = jsonb_set(
                  jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{ingestion,attempts}',
                    COALESCE(metadata->'ingestion'->'attempts', '[]'::jsonb) || to_jsonb($3::text),
                    true
                  ),
                  '{ingestion,last_attempt_id}', to_jsonb($3::text), true
                ),
                updated_at = now()
                WHERE tenant_id = $1 AND doc_id = $2::uuid
                """,
                ctx.tenant_id,
                str(doc_id),
                str(attempt_id),
            )

    async def end_attempt(self, *, ctx: TenantContext, doc_id: UUID, attempt_id: UUID, success: bool) -> None:
        """Registra fin de intento de ingesta y resultado en documents.metadata."""
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", ctx.tenant_id)
            await conn.execute(
                """
                UPDATE documents
                SET metadata = jsonb_set(
                  jsonb_set(
                    COALESCE(metadata, '{}'::jsonb),
                    '{ingestion,last_attempt_status}', to_jsonb(CASE WHEN $3 THEN 'success' ELSE 'failed' END), true
                  ),
                  '{ingestion,last_attempt_at}', to_jsonb(now()::text), true
                ),
                updated_at = now()
                WHERE tenant_id = $1 AND doc_id = $2::uuid
                """,
                ctx.tenant_id,
                str(doc_id),
                success,
            )
