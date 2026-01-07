from dataclasses import dataclass
from typing import Any
from uuid import UUID

from packages.shared.schemas.chat import ChatFilters
from packages.shared.schemas.common import TenantContext
from ports import PermissionsService, Reranker, VectorStore


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: UUID
    title: str
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    evidence_strength: float
    debug: dict[str, Any]


class RetrievalService:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        permissions: PermissionsService,
        reranker: Reranker | None = None,
    ):
        self._vector_store = vector_store
        self._permissions = permissions
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        ctx: TenantContext,
        question: str,
        filters: ChatFilters | None,
        top_k: int,
        use_rerank: bool,
    ) -> RetrievalResult:
        meta_filters: dict[str, Any] = {"tenant_id": str(ctx.tenant_id)}

        if filters:
            if filters.department:
                meta_filters["department"] = filters.department
            if filters.doc_type:
                meta_filters["doc_type"] = filters.doc_type
            if filters.tags:
                meta_filters["tags"] = {"$contains_any": filters.tags}
            if filters.date_from:
                meta_filters["date_from"] = filters.date_from.isoformat()
            if filters.date_to:
                meta_filters["date_to"] = filters.date_to.isoformat()

        perm_filters: dict[str, Any] = await self._permissions.vector_filters_for(ctx)
        meta_filters.update(perm_filters)

        candidates: list[RetrievedChunk] = await self._vector_store.search(
            tenant_id=ctx.tenant_id,
            query=question,
            top_k=top_k,
            filters=meta_filters,
        )

        chunks = candidates
        if use_rerank and self._reranker and candidates:
            chunks = await self._reranker.rerank(question=question, chunks=candidates)

        selected = chunks[: min(6, len(chunks))]

        strength = (sum(c.score for c in selected) / len(selected)) if selected else 0.0

        debug: dict[str, Any] = {
            "top_k": top_k,
            "selected_n": len(selected),
            "filters": meta_filters,
            "scores": [c.score for c in selected],
            "doc_ids": [str(c.doc_id) for c in selected],
        }
        return RetrievalResult(chunks=selected, evidence_strength=strength, debug=debug)
