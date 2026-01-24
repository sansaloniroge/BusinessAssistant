from __future__ import annotations

from typing import Any, Protocol, Sequence, TYPE_CHECKING

from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.filters import MetaFilters

if TYPE_CHECKING:
    from uuid import UUID
    from .retrieval_service import RetrievedChunk


class VectorStore(Protocol):
    async def search_by_embedding(
        self,
        *,
        tenant_id: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: MetaFilters,
    ) -> list[RetrievedChunk]:
        ...

    async def upsert_chunks(self, *, tenant_id: str, chunks: Sequence[dict[str, Any]]) -> int:
        """Inserta/actualiza chunks. Devuelve nº de filas afectadas."""
        ...

    async def delete_by_doc_id(self, *, tenant_id: str, doc_id: str) -> int:
        """Borra todos los chunks de un doc. Devuelve nº de filas borradas."""
        ...

    async def health(self) -> bool:
        ...


class EmbeddingService(Protocol):
    async def embed_query(self, *, text: str) -> list[float]:
        ...


class PermissionsService(Protocol):
    async def vector_filters_for(self, ctx: TenantContext) -> MetaFilters:
        ...

    async def can_access_doc(self, *, ctx: TenantContext, doc_id: "UUID") -> bool:
        ...


class Reranker(Protocol):
    async def rerank(self, *, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...
