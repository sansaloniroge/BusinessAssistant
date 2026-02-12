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

    # Nuevo debug: separación de filtros
    assert res.debug["base_filters"]["tenant_id"] == str(tenant_ctx.tenant_id)
    assert "department" in res.debug["base_filters"]
    assert "acl" in res.debug["perm_filters"]
    assert "acl" in res.debug["effective_filters"]


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


@pytest.mark.asyncio
async def test_retrieve_caps_chunks_per_doc_and_total_limit(tenant_ctx):
    doc_a = uuid4()
    doc_b = uuid4()
    doc_c = uuid4()

    # Orden ya viene por score/orden de candidates; el selector debe:
    # - coger como mucho 2 del doc_a aunque haya 3
    # - respetar el orden
    chunks = [
        RetrievedChunk(chunk_id="a1", doc_id=doc_a, title="A", content="x", score=0.99, metadata={}),
        RetrievedChunk(chunk_id="a2", doc_id=doc_a, title="A", content="x", score=0.98, metadata={}),
        RetrievedChunk(chunk_id="a3", doc_id=doc_a, title="A", content="x", score=0.97, metadata={}),
        RetrievedChunk(chunk_id="b1", doc_id=doc_b, title="B", content="y", score=0.96, metadata={}),
        RetrievedChunk(chunk_id="b2", doc_id=doc_b, title="B", content="y", score=0.95, metadata={}),
        RetrievedChunk(chunk_id="c1", doc_id=doc_c, title="C", content="z", score=0.94, metadata={}),
        # uno más para forzar límite total si se extendiera
        RetrievedChunk(chunk_id="c2", doc_id=doc_c, title="C", content="z", score=0.93, metadata={}),
    ]

    svc = RetrievalService(vector_store=_VectorStore(chunks), permissions=_Perms(), embeddings=_Emb())

    res = await svc.retrieve(ctx=tenant_ctx, question="q", filters=None, top_k=50, use_rerank=False)

    # cap por doc: a3 no puede entrar
    assert [c.chunk_id for c in res.chunks].count("a1") == 1
    assert [c.chunk_id for c in res.chunks].count("a2") == 1
    assert "a3" not in [c.chunk_id for c in res.chunks]

    # total limit: 6 máximo
    assert len(res.chunks) <= svc.MAX_SELECTED_CHUNKS
    assert res.debug["selected_n"] == len(res.chunks)

    # y debe preservar el orden relativo esperado (a1, a2, b1, b2, c1, c2)
    assert [c.chunk_id for c in res.chunks] == ["a1", "a2", "b1", "b2", "c1", "c2"]
