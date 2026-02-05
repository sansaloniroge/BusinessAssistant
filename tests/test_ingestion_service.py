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


@pytest.mark.asyncio
async def test_ingestion_service_state_transitions():
    conn = _Conn()
    svc = IngestionService(pool=_Pool(conn))
    ctx = TenantContext(tenant_id="t1", user_id=uuid4(), role="user", scopes=[])
    doc_id = uuid4()

    # pending -> processing
    await svc.start_processing(ctx=ctx, doc_id=doc_id)
    # processing -> ready
    await svc.mark_ready(ctx=ctx, doc_id=doc_id)
    # any -> failed
    await svc.mark_failed(ctx=ctx, doc_id=doc_id)

    # Verify RLS set_config called and updates executed
    statements = [sql for sql, _ in conn.executed]
    assert any("set_config('app.tenant_id'" in sql for sql in statements)
    assert any(sql.startswith("UPDATE documents") and "SET status = 'processing'" in sql for sql in statements)
    assert any(sql.startswith("UPDATE documents") and "SET status = 'ready'" in sql for sql in statements)
    assert any(sql.startswith("UPDATE documents") and "SET status = 'failed'" in sql for sql in statements)

