from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from packages.shared.schemas.chat import ChatFilters
from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.filters import FieldFilter, MetaFilters

from apps.api.services.ports import EmbeddingService, PermissionsService, Reranker, VectorStore


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
    DATE_FIELD = "doc_date"

    def __init__(
        self,
        vector_store: VectorStore,
        permissions: PermissionsService,
        embeddings: EmbeddingService,
        reranker: Reranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._permissions = permissions
        self._embeddings = embeddings
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

        meta_filters.update(self._chat_filters_to_meta(filters))

        perm_filters: dict[str, Any] = await self._permissions.vector_filters_for(ctx)
        meta_filters.update(perm_filters)

        q_emb = await self._embeddings.embed_query(text=question)

        candidates: list[RetrievedChunk] = await self._vector_store.search_by_embedding(
            tenant_id=str(ctx.tenant_id),
            query_embedding=q_emb,
            top_k=top_k,
            filters=meta_filters,
        )

        # 6) optional rerank
        chunks = candidates
        reranked = False
        if use_rerank and self._reranker and candidates:
            chunks = await self._reranker.rerank(question=question, chunks=candidates)
            reranked = True

        selected = chunks[: min(6, len(chunks))]
        strength = (sum(c.score for c in selected) / len(selected)) if selected else 0.0

        debug: dict[str, Any] = {
            "top_k": top_k,
            "selected_n": len(selected),
            "reranked": reranked,
            "filters": meta_filters,
            "scores": [c.score for c in selected],
            "doc_ids": [str(c.doc_id) for c in selected],
            "chunk_ids": [c.chunk_id for c in selected],
        }
        return RetrievalResult(chunks=selected, evidence_strength=strength, debug=debug)

    def _chat_filters_to_meta(self, filters: ChatFilters | None) -> MetaFilters:
        meta: MetaFilters = {}
        if not filters:
            return meta

        if filters.department:
            meta["department"] = FieldFilter(op="$eq", value=filters.department)

        if filters.doc_type:
            meta["doc_type"] = FieldFilter(op="$eq", value=filters.doc_type)

        if filters.tags:
            meta["tags"] = FieldFilter(op="$contains_any", value=filters.tags)

        if filters.date_from:
            meta[f"{self.DATE_FIELD}__gte"] = FieldFilter(
                op="$gte", value=self._to_utc_iso(filters.date_from)
            )

        if filters.date_to:
            meta[f"{self.DATE_FIELD}__lte"] = FieldFilter(
                op="$lte", value=self._to_utc_iso(filters.date_to)
            )

        return meta

    @staticmethod
    def _to_utc_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
