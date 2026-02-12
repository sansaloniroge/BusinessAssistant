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

    # retrieval limits (V1)
    MAX_SELECTED_CHUNKS: int = 6
    MAX_CHUNKS_PER_DOC: int = 2

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
        # 1) Filtros base de producto (siempre tenant)
        base_filters: dict[str, Any] = {"tenant_id": str(ctx.tenant_id)}
        base_filters.update(self._chat_filters_to_meta(filters))

        # 2) Filtros de permisos (siempre)
        perm_filters: dict[str, Any] = await self._permissions.vector_filters_for(ctx)

        # 3) Filtros efectivos
        effective_filters: dict[str, Any] = {**base_filters, **perm_filters}

        q_emb = await self._embeddings.embed_query(text=question)

        candidates: list[RetrievedChunk] = await self._vector_store.search_by_embedding(
            tenant_id=str(ctx.tenant_id),
            query_embedding=q_emb,
            top_k=top_k,
            filters=effective_filters,
        )

        # 4) optional rerank (controlado)
        chunks = candidates
        reranked = False
        if use_rerank and self._reranker and candidates:
            chunks = await self._reranker.rerank(question=question, chunks=candidates)
            reranked = True

        # 5) Selección “seria”: diversidad + límites explícitos
        selected = self._select_diverse(chunks)
        strength = (sum(c.score for c in selected) / len(selected)) if selected else 0.0

        debug: dict[str, Any] = {
            "top_k": top_k,
            "candidate_n": len(candidates),
            "selected_n": len(selected),
            "reranked": reranked,
            "base_filters": base_filters,
            "perm_filters": perm_filters,
            "effective_filters": effective_filters,
            "candidate_scores": [c.score for c in candidates[: min(len(candidates), 20)]],
            "scores": [c.score for c in selected],
            "doc_ids": [str(c.doc_id) for c in selected],
            "doc_ids_unique": list(dict.fromkeys(str(c.doc_id) for c in selected)),
            "chunk_ids": [c.chunk_id for c in selected],
        }
        return RetrievalResult(chunks=selected, evidence_strength=strength, debug=debug)

    def _select_diverse(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Selecciona chunks con diversidad por doc_id y límites explícitos.

        Estrategia V1:
        - Mantener el orden actual (ya vector o rerank).
        - Cap por documento (MAX_CHUNKS_PER_DOC).
        - Total máximo (MAX_SELECTED_CHUNKS).
        """
        if not chunks:
            return []

        per_doc: dict[str, int] = {}
        out: list[RetrievedChunk] = []

        for c in chunks:
            doc_key = str(c.doc_id)
            if per_doc.get(doc_key, 0) >= self.MAX_CHUNKS_PER_DOC:
                continue
            out.append(c)
            per_doc[doc_key] = per_doc.get(doc_key, 0) + 1
            if len(out) >= self.MAX_SELECTED_CHUNKS:
                break

        return out

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
