import pytest

from apps.api.adapters.pgvector_vector_store import PgvectorVectorStore


class _Conn:
    def __init__(self, existing_rows):
        self._existing_rows = existing_rows
        self.executed = []
        self.executemany_calls = []
        self.fetched = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self.fetched.append((sql, args))
        return list(self._existing_rows)

    async def executemany(self, sql, records):
        self.executemany_calls.append((sql, list(records)))
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
async def test_upsert_chunks_fail_fast_on_embedding_model_or_chunker_version_mismatch():
    tenant_id = "t1"

    # Chunk ya existe en DB con otra combinación
    existing = [
        {
            "chunk_id": "c1",
            "embedding_model": "old-model",
            "chunker_version": "v1",
        }
    ]
    conn = _Conn(existing_rows=existing)
    store = PgvectorVectorStore(pool=_Pool(conn))

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "00000000-0000-0000-0000-000000000001",
            "title": "T",
            "content": "x",
            "embedding": [0.0] * 3,
            "metadata": {},
            "embedding_model": "new-model",
            "chunker_version": "v1",
        }
    ]

    with pytest.raises(RuntimeError) as e:
        await store.upsert_chunks(tenant_id=tenant_id, chunks=chunks)

    assert "fail-fast" in str(e.value)

    # No debe intentar insertar/actualizar si falla
    assert conn.executemany_calls == []


@pytest.mark.asyncio
async def test_upsert_chunks_requires_embedding_model_and_chunker_version():
    conn = _Conn(existing_rows=[])
    store = PgvectorVectorStore(pool=_Pool(conn))

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "00000000-0000-0000-0000-000000000001",
            "title": "T",
            "content": "x",
            "embedding": [0.0] * 3,
            "metadata": {},
            "embedding_model": "",
            "chunker_version": "v1",
        }
    ]

    with pytest.raises(ValueError):
        await store.upsert_chunks(tenant_id="t1", chunks=chunks)

