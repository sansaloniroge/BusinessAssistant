import pytest
from uuid import uuid4

from apps.api.services.ingestion_service import IngestionService
from packages.shared.schemas.common import TenantContext


class _Conn:
    def __init__(self):
        self.executed = []

    async def execute(self, sql, *args):
        self.executed.append((sql.strip(), args))
        return "OK"


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


def test_stable_chunk_id_is_deterministic_and_unique_per_index():
    doc_id = uuid4()
    c1 = IngestionService.stable_chunk_id(doc_id=doc_id, index=0)
    c2 = IngestionService.stable_chunk_id(doc_id=doc_id, index=0)
    c3 = IngestionService.stable_chunk_id(doc_id=doc_id, index=1)

    assert c1 == c2
    assert c1 != c3
    # UUIDv5 string format
    assert isinstance(c1, str) and len(c1) == 36


@pytest.mark.asyncio
async def test_start_end_attempt_track_metadata_ingestion():
    conn = _Conn()
    svc = IngestionService(pool=_Pool(conn))
    ctx = TenantContext(tenant_id="t1", user_id=uuid4(), role="user", scopes=[])
    doc_id = uuid4()
    attempt_id = uuid4()

    await svc.start_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id)
    await svc.end_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id, success=True)

    statements = [sql for sql, _ in conn.executed]
    # set_config must be called before each update
    assert any("set_config('app.tenant_id'" in sql for sql in statements)
    # jsonb_set for attempts and last_attempt_id
    assert any("jsonb_set" in sql and "{ingestion,attempts}" in sql for sql in statements)
    assert any("jsonb_set" in sql and "{ingestion,last_attempt_id}" in sql for sql in statements)
    # end_attempt updates status and timestamp
    assert any("{ingestion,last_attempt_status}" in sql for sql in statements)
    assert any("{ingestion,last_attempt_at}" in sql for sql in statements)

