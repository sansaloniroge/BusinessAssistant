from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from packages.shared.schemas.chat import ChatFilters
from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.filters import FieldFilter
from packages.shared.schemas.filters import MetaFilters
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
    DATE_FIELD = "doc_date"
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
            top_k: int = 8,
            use_rerank: bool = False,
    ) -> RetrievalResult:
        # 1) Meta filters from user-facing ChatFilters
        meta_filters: MetaFilters = self._chat_filters_to_meta(filters)

        # 2) Permissions filters (ACL / scopes / roles), merged in
        perm_filters = await self._permissions.vector_filters_for(ctx)

        # 3) Tenant isolation ALWAYS applied
        # (tenant_id is included as a hard filter)
        final_filters: dict[str, Any] = {"tenant_id": ctx.tenant_id, **meta_filters, **perm_filters}

        # 4) Vector search (current contract: query is text)
        candidates = await self._vector_store.search(
            tenant_id=ctx.tenant_id,
            query=question,
            top_k=top_k,
            filters=final_filters,
        )

        # 5) Optional rerank (if enabled + reranker available)
        reranked = False
        if use_rerank and self._reranker and candidates:
            candidates = await self._reranker.rerank(question=question, chunks=candidates)
            reranked = True

        # 6) Select final chunks (cap)
        selected = candidates[:6]

        # 7) Evidence strength: mean score (0.0 if none)
        if selected:
            evidence_strength = sum(c.score for c in selected) / float(len(selected))
        else:
            evidence_strength = 0.0

        # 8) Debug payload for observability / tracing
        debug: dict[str, Any] = {
            "top_k": top_k,
            "candidate_n": len(candidates),
            "selected_n": len(selected),
            "reranked": reranked,
            "filters": final_filters,
            "scores": [c.score for c in selected],
            "doc_ids": [str(c.doc_id) for c in selected],
            "chunk_ids": [c.chunk_id for c in selected],
        }

        return RetrievalResult(chunks=selected, evidence_strength=evidence_strength, debug=debug)

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
                op="$gte",
                value=self._to_utc_iso(filters.date_from),
            )

        if filters.date_to:
            meta[f"{self.DATE_FIELD}__lte"] = FieldFilter(
                op="$lte",
                value=self._to_utc_iso(filters.date_to),
            )

        return meta

    @staticmethod
    def _to_utc_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()