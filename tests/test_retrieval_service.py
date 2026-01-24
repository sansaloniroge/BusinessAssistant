import pytest
from uuid import uuid4

from apps.api.services.retrieval_service import RetrievedChunk, RetrievalService
from packages.shared.schemas.chat import ChatFilters
from packages.shared.schemas.filters import FieldFilter


class _VectorStore:
    def __init__(self, chunks):
        self._chunks = chunks
        self.last_call = None

    async def search_by_embedding(self, *, tenant_id, query_embedding, top_k, filters):
        self.last_call = {
            "tenant_id": tenant_id,
            "query_embedding": query_embedding,
            "top_k": top_k,
            "filters": filters,
        }
        return self._chunks


class _Perms:
    def __init__(self, extra=None):
        self._extra = extra or {}

    async def vector_filters_for(self, ctx):
        return dict(self._extra)


class _Emb:
    async def embed_query(self, *, text):
        return [0.1, 0.2, 0.3]


class _Reranker:
    def __init__(self, reordered):
        self._reordered = reordered

    async def rerank(self, *, question, chunks):
        return self._reordered


@pytest.mark.asyncio
async def test_retrieve_merges_filters_and_computes_strength(tenant_ctx):
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id=uuid4(),
            title="T1",
            content="x",
            score=0.5,
            metadata={},
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id=uuid4(),
            title="T2",
            content="y",
            score=0.3,
            metadata={},
        ),
    ]

    vs = _VectorStore(chunks)
    svc = RetrievalService(vector_store=vs, permissions=_Perms({"acl": FieldFilter(op="$eq", value="ok")}), embeddings=_Emb())

    filters = ChatFilters(department="HR", tags=["a", "b"])

    res = await svc.retrieve(ctx=tenant_ctx, question="q", filters=filters, top_k=12, use_rerank=False)

    assert res.chunks == chunks[:2]
    assert res.evidence_strength == pytest.approx((0.5 + 0.3) / 2)

    merged = vs.last_call["filters"]
    assert merged["tenant_id"] == str(tenant_ctx.tenant_id)
    assert isinstance(merged["department"], FieldFilter)
    assert merged["department"].op == "$eq"
    assert merged["tags"].op == "$contains_any"
    assert merged["acl"].op == "$eq"


@pytest.mark.asyncio
async def test_retrieve_reranks_when_enabled(tenant_ctx):
    base = [
        RetrievedChunk(chunk_id="c1", doc_id=uuid4(), title="T1", content="x", score=0.1, metadata={}),
        RetrievedChunk(chunk_id="c2", doc_id=uuid4(), title="T2", content="y", score=0.9, metadata={}),
    ]
    reordered = [base[1], base[0]]

    svc = RetrievalService(
        vector_store=_VectorStore(base),
        permissions=_Perms(),
        embeddings=_Emb(),
        reranker=_Reranker(reordered),
    )

    res = await svc.retrieve(ctx=tenant_ctx, question="q", filters=None, top_k=2, use_rerank=True)
    assert res.chunks[0].chunk_id == "c2"
    assert res.debug["reranked"] is True

