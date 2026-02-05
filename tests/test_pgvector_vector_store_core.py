import pytest

from apps.api.adapters.pgvector_vector_store import PgvectorVectorStore


class _Conn:
    def __init__(self, rows=None, fetchval=1):
        self._rows = rows or []
        self._executed = []
        self._fetchval = fetchval
        self._delete_result = "DELETE 0"

    async def execute(self, sql, *args):
        self._executed.append((sql, args))
        # emulate DELETE n
        if sql.strip().startswith("DELETE"):
            return self._delete_result
        return "OK"

    async def fetch(self, sql, *args):
        return list(self._rows)

    async def fetchval(self, sql, *args):
        return self._fetchval

    async def executemany(self, sql, records):
        # no-op for tests
        return None


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


@pytest.mark.asyncio
async def test_search_by_embedding_filters_and_score_ordering():
    # Prepare fake rows in the shape asyncpg Record would be accessed by keys
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "00000000-0000-0000-0000-000000000001",
            "title": "T1",
            "content": "x1",
            "metadata": {"section": "A"},
            "score": 0.9,
        },
        {
            "chunk_id": "c2",
            "doc_id": "00000000-0000-0000-0000-000000000002",
            "title": "T2",
            "content": "x2",
            "metadata": {"section": "B"},
            "score": 0.7,
        },
    ]
    conn = _Conn(rows=rows)
    store = PgvectorVectorStore(pool=_Pool(conn))

    filters = {
        "department": {"op": "$eq", "value": "HR"},
        "doc_type": {"op": "$eq", "value": "policy"},
        "tags": {"op": "$contains_any", "value": ["a", "b"]},
        "doc_date__gte": {"op": "$gte", "value": "2020-01-01T00:00:00Z"},
        "doc_date__lte": {"op": "$lte", "value": "2025-01-01T00:00:00Z"},
        "language": {"op": "$eq", "value": "es"},
        "audience": {"op": "$in", "value": ["staff", "hr"]},
    }

    res = await store.search_by_embedding(
        tenant_id="t1",
        query_embedding=[0.0] * PgvectorVectorStore.EMBEDDING_DIM,
        top_k=5,
        filters=filters,
    )

    assert len(res) == 2
    # Score propagated
    assert res[0].score >= res[1].score


@pytest.mark.asyncio
async def test_upsert_happy_path_inserts_records():
    conn = _Conn()
    store = PgvectorVectorStore(pool=_Pool(conn))

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "00000000-0000-0000-0000-000000000001",
            "title": "T",
            "content": "x",
            "embedding": [0.0] * PgvectorVectorStore.EMBEDDING_DIM,
            "metadata": {},
            "embedding_model": "model-1",
            "chunker_version": "v1",
        }
    ]

    n = await store.upsert_chunks(tenant_id="t1", chunks=chunks)
    assert n == 1


@pytest.mark.asyncio
async def test_delete_by_doc_id_returns_count():
    conn = _Conn()
    conn._delete_result = "DELETE 3"
    store = PgvectorVectorStore(pool=_Pool(conn))

    n = await store.delete_by_doc_id(tenant_id="t1", doc_id="00000000-0000-0000-0000-000000000001")
    assert n == 3


@pytest.mark.asyncio
async def test_delete_by_chunk_id_returns_count():
    conn = _Conn()
    conn._delete_result = "DELETE 1"
    store = PgvectorVectorStore(pool=_Pool(conn))

    n = await store.delete_by_chunk_id(tenant_id="t1", chunk_id="c1")
    assert n == 1


@pytest.mark.asyncio
async def test_health_returns_true():
    conn = _Conn(fetchval=1)
    store = PgvectorVectorStore(pool=_Pool(conn))
    assert await store.health() is True

