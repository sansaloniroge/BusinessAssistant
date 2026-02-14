from __future__ import annotations

import os
from typing import Sequence
from uuid import UUID

import asyncpg
from fastapi import Depends, Header, HTTPException

from packages.shared.schemas.common import TenantContext

from apps.api.adapters.pgvector_vector_store import PgvectorVectorStore
from apps.api.services.chat_service import ChatService
from apps.api.services.citation_service import CitationService
from apps.api.services.llm_client import LLMClient
from apps.api.services.observability import setup_observability
from apps.api.services.permissions import DefaultPermissionsService
from apps.api.services.prompt_service import PromptService
from apps.api.services.retrieval_service import RetrievalService
from apps.api.services.run_logger import RunLogger

from apps.api.services.ports import EmbeddingService


def setup_app(app) -> None:
    # Observabilidad (OTLP si está configurado por env)
    setup_observability()


async def get_db_pool() -> asyncpg.Pool:
    # Singleton por proceso
    # Nota: en tests/CLI se puede sobreescribir con dependency overrides.
    if not hasattr(get_db_pool, "_pool"):
        dsn = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/businessassistant")
        get_db_pool._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)  # type: ignore[attr-defined]
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


async def get_chat_service(pool: asyncpg.Pool = Depends(get_db_pool)) -> ChatService:
    # Infra/adapters mínimos
    vector_store = PgvectorVectorStore(pool)
    permissions = DefaultPermissionsService()

    # Embeddings/LLM: placeholder (hasta integrar proveedor real)
    llm = LLMClient()

    retrieval = RetrievalService(vector_store=vector_store, permissions=permissions, embeddings=_DummyEmbeddings())

    # Repos mínimos (runs/conversations/messages): en V1 usamos repos in-memory si no hay DB tables.
    runs_repo = _InMemoryRunsRepo()
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
class _DummyEmbeddings(EmbeddingService):
    async def embed_query(self, *, text: str) -> list[float]:
        # TODO: integrar provider de embeddings.
        raise HTTPException(status_code=501, detail="Embeddings provider not configured")

    async def embed_chunks(self, *, texts: Sequence[str]) -> list[Sequence[float]]:
        # Firma exacta: `texts: Sequence[str]` en el protocolo.
        raise HTTPException(status_code=501, detail="Embeddings provider not configured")


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
