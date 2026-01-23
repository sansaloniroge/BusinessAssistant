from __future__ import annotations

from typing import Any, Protocol, Sequence, TYPE_CHECKING

from packages.shared.schemas.common import TenantContext

if TYPE_CHECKING:
    from .retrieval_service import RetrievedChunk


class VectorStore(Protocol):
    async def search_by_embedding(
        self,
        *,
        tenant_id: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievedChunk]:
        ...


class EmbeddingService(Protocol):
    async def embed_query(self, *, text: str) -> list[float]:
        ...


class PermissionsService(Protocol):
    async def vector_filters_for(self, ctx: TenantContext) -> dict[str, Any]:
        ...


class Reranker(Protocol):
    async def rerank(self, *, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...
