from __future__ import annotations

import os
from typing import Sequence, cast
from uuid import UUID

import asyncpg
from fastapi import Depends, Header, HTTPException

from packages.shared.schemas.common import TenantContext

from apps.api.adapters.pgvector_vector_store import PgvectorVectorStore
from apps.api.services.chat_service import ChatService
from apps.api.services.citation_service import CitationService
from apps.api.services.llm_client import LLMClient
from apps.api.services.llm_client import LLMResult
from apps.api.services.observability import setup_observability
from apps.api.services.permissions import DefaultPermissionsService
from apps.api.services.prompt_service import PromptService
from apps.api.services.retrieval_service import RetrievalService
from apps.api.services.run_logger import RunLogger

from apps.api.services.ports import EmbeddingService


def setup_app(app) -> None:
    # Observabilidad (OTLP si está configurado por env)
    setup_observability()


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Asegura que RLS se aplica incluso si el rol tuviera BYPASSRLS.
    # (equivalente a: SET row_security = on)
    await conn.execute("SET row_security = on")


async def get_db_pool() -> asyncpg.Pool:
    # Singleton por proceso
    # Nota: en tests/CLI se puede sobreescribir con dependency overrides.
    if not hasattr(get_db_pool, "_pool"):
        dsn = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/businessassistant")
        get_db_pool._pool = await asyncpg.create_pool(  # type: ignore[attr-defined]
            dsn=dsn,
            min_size=1,
            max_size=10,
            init=_init_connection,
        )
    return get_db_pool._pool  # type: ignore[attr-defined]


async def get_ctx(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_role: str | None = Header(default="user", alias="X-Role"),
    x_scopes: str | None = Header(default="", alias="X-Scopes"),
) -> TenantContext:
    """Resolver provisional de TenantContext.

    Hasta implementar AuthN/Z (paso 8), aceptamos contexto por headers.
    """
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-Id")
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id")

    try:
        user_id = UUID(x_user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid X-User-Id (must be UUID)")

    scopes = [s.strip() for s in (x_scopes or "").split(",") if s.strip()]

    return TenantContext(
        tenant_id=str(x_tenant_id),
        user_id=user_id,
        role=str(x_role or "user"),
        scopes=scopes,
    )


def _is_dev_mode() -> bool:
    return os.getenv("APP_ENV", os.getenv("ENV", "dev")).lower() in {"dev", "local"}


class _PostgresRunsRepo:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert_run(self, *, run_id, data) -> None:
        # Persistencia mínima viable para fase 7 (eval/judge): estable y consultable.
        sql = """
        INSERT INTO runs (
          tenant_id, run_id, user_id, conversation_id,
          question, answer, model,
          usage, retrieved_doc_ids, retrieval_debug
        ) VALUES (
          $1, $2::uuid, $3::uuid, $4::uuid,
          $5, $6, $7,
          $8::jsonb, $9::jsonb, $10::jsonb
        )
        """

        usage = getattr(data, "usage", None)
        usage_json = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None

        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(data.tenant_id))
            await conn.execute(
                sql,
                str(data.tenant_id),
                str(run_id),
                str(data.user_id),
                str(data.conversation_id),
                str(data.question),
                str(data.answer),
                str(data.model),
                usage_json,
                list(getattr(data, "retrieved_doc_ids", []) or []),
                dict(getattr(data, "retrieval_debug", {}) or {}),
            )


async def get_chat_service(pool: asyncpg.Pool = Depends(get_db_pool)) -> ChatService:
    # Infra/adapters mínimos
    vector_store = PgvectorVectorStore(pool)
    permissions = DefaultPermissionsService()

    # DEV: stubs para poder probar flujo HTTP sin proveedores externos.
    if _is_dev_mode() and os.getenv("DEV_DUMMY_LLM", "true").lower() in {"1", "true", "yes"}:
        llm: LLMClient = _DummyLLM()
    else:
        llm = LLMClient()

    if _is_dev_mode() and os.getenv("DEV_DUMMY_EMBEDDINGS", "true").lower() in {"1", "true", "yes"}:
        embeddings = cast(EmbeddingService, _DummyEmbeddings())
    else:
        embeddings = cast(EmbeddingService, _DummyEmbeddings())

    retrieval = RetrievalService(vector_store=vector_store, permissions=permissions, embeddings=embeddings)

    runs_repo = _PostgresRunsRepo(pool)
    conversations_repo = _InMemoryConversationsRepo()
    messages_repo = _InMemoryMessagesRepo()

    return ChatService(
        retrieval=retrieval,
        prompts=PromptService(),
        llm=llm,
        citations=CitationService(),
        run_logger=RunLogger(runs_repo),
        conversations_repo=conversations_repo,
        messages_repo=messages_repo,
    )


# === Placeholders simples para poder levantar API antes de tener capa repo completa ===
class _DummyEmbeddings:
    """Embeddings dummy (DEV).

    Ojo: debe devolver dimensión 1536 para que el vector store no haga fail-fast.
    """

    async def embed_query(self, *, text: str) -> list[float]:
        # Vector determinista y barato (no semántico).
        # Mantener 1536 floats.
        base = float((abs(hash(text)) % 10_000) / 10_000)
        return [base] * 1536

    async def embed_chunks(self, *, texts: Sequence[str]) -> list[Sequence[float]]:
        return [[float((abs(hash(t)) % 10_000) / 10_000)] * 1536 for t in texts]


class _DummyLLM(LLMClient):
    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        import asyncio

        # Simula latencia para que métricas/evals no tengan casos "0ms".
        await asyncio.sleep(0.02)

        # Respuesta simple que siempre cita el primer chunk si existe.
        text = f"(dev) No tengo un LLM configurado. Pregunta: {user}\n\n[C1]"

        # Usage coherente (aunque sea dummy). La latencia real la añade LLMClient.generate().
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_estimate_usd": 0.0}
        return LLMResult(text=text, usage=usage)


class _InMemoryRunsRepo:
    def __init__(self) -> None:
        self._items = []

    async def insert_run(self, *, run_id, data):
        self._items.append((run_id, data))


class _InMemoryConversationsRepo:
    def __init__(self) -> None:
        self._items = set()

    async def create(self, conversation_id, ctx):
        self._items.add((conversation_id, str(ctx.tenant_id)))


class _InMemoryMessagesRepo:
    def __init__(self) -> None:
        self._items = []

    async def insert(self, conversation_id, role, content):
        self._items.append((conversation_id, role, content))
