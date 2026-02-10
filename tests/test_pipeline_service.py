import pytest
from uuid import uuid4

from packages.shared.schemas.common import TenantContext

from apps.api.services.pipeline_service import PipelineService


class _IngestionSpy:
    def __init__(self):
        self.calls = []

    async def start_processing(self, *, ctx, doc_id):
        self.calls.append(("start_processing", doc_id))

    async def start_attempt(self, *, ctx, doc_id, attempt_id):
        self.calls.append(("start_attempt", doc_id, attempt_id))

    async def mark_ready(self, *, ctx, doc_id):
        self.calls.append(("mark_ready", doc_id))

    async def mark_failed(self, *, ctx, doc_id):
        self.calls.append(("mark_failed", doc_id))

    async def end_attempt(self, *, ctx, doc_id, attempt_id, success: bool):
        self.calls.append(("end_attempt", doc_id, attempt_id, success))

    @staticmethod
    def stable_chunk_id(*, doc_id, index: int) -> str:
        # Determinista para test
        return f"{doc_id}:{index}"


class _Extractor:
    async def extract(self, *, tenant_id: str, doc_id):
        return b"Hello world"


class _Loader:
    async def load(self, *, raw, metadata):
        return {"title": "Doc", "text": raw.decode("utf-8"), "metadata": metadata}


class _Normalizer:
    async def normalize(self, *, doc):
        return doc


class _Chunker:
    async def chunk(self, *, doc):
        # Genera 2 chunks
        return [
            {"index": 0, "title": doc.get("title"), "content": doc.get("text"), "metadata": {"section": "A"}},
            {"index": 1, "title": doc.get("title"), "content": doc.get("text"), "metadata": {"section": "B"}},
        ]


class _Embeddings:
    async def embed_chunks(self, *, texts):
        # Devuelve vectores de 1536 elementos
        return [[0.0] * 1536 for _ in texts]


class _VectorStoreSpy:
    def __init__(self):
        self.upserts = []

    async def upsert_chunks(self, *, tenant_id: str, chunks):
        self.upserts.append((tenant_id, chunks))
        return len(chunks)


class _MetricsSinkSpy:
    def __init__(self):
        self.records = []

    async def record(self, *, tenant_id: str, doc_id, stage: str, metrics: dict):
        self.records.append((stage, metrics))


class _DLQSpy:
    def __init__(self):
        self.sent = []

    async def send(self, *, tenant_id: str, doc_id, reason: str, payload: dict):
        self.sent.append({"tenant_id": tenant_id, "doc_id": doc_id, "reason": reason, "payload": payload})


@pytest.mark.asyncio
async def test_pipeline_happy_path_records_metrics_and_marks_ready():
    ctx = TenantContext(tenant_id="t1", user_id=uuid4(), role="user", scopes=[])
    doc_id = uuid4()

    ingestion = _IngestionSpy()
    extractor = _Extractor()
    loader = _Loader()
    normalizer = _Normalizer()
    chunker = _Chunker()
    embeddings = _Embeddings()
    vector_store = _VectorStoreSpy()
    metrics = _MetricsSinkSpy()

    svc = PipelineService(
        ingestion=ingestion,
        extractor=extractor,
        loader=loader,
        normalizer=normalizer,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
        metrics=metrics,
        embedding_model="text-embedding-3-small",
        chunker_version="v1",
        max_retries=0,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )

    await svc.run(ctx=ctx, doc_id=doc_id, base_metadata={"department": "HR", "doc_type": "policy"})

    stages = [s for s, _ in metrics.records]
    assert stages == ["extract", "load", "normalize", "chunk", "embed", "upsert", "attempt"]

    # Debe marcar ready y cerrar intento con success
    assert any(c[0] == "mark_ready" for c in ingestion.calls)
    assert any(c[0] == "end_attempt" and c[-1] is True for c in ingestion.calls)

    # VectorStore recibió 2 chunks con metadatos y modelo
    assert len(vector_store.upserts) == 1
    tenant_id, chunks = vector_store.upserts[0]
    assert tenant_id == "t1"
    assert len(chunks) == 2
    for i, c in enumerate(chunks):
        assert c["embedding_model"] == "text-embedding-3-small"
        assert c["chunker_version"] == "v1"
        assert c["doc_id"] == doc_id
        assert c["chunk_id"].endswith(f":{i}")
        assert c["department"] == "HR"
        assert c["doc_type"] == "policy"


@pytest.mark.asyncio
async def test_pipeline_transient_failure_retries_then_succeeds_and_does_not_mark_failed():
    class _FlakyExtractor:
        def __init__(self):
            self.calls = 0

        async def extract(self, *, tenant_id: str, doc_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return b"Hello world"

    ctx = TenantContext(tenant_id="t1", user_id=uuid4(), role="user", scopes=[])
    doc_id = uuid4()

    ingestion = _IngestionSpy()
    extractor = _FlakyExtractor()
    loader = _Loader()
    normalizer = _Normalizer()
    chunker = _Chunker()
    embeddings = _Embeddings()
    vector_store = _VectorStoreSpy()
    metrics = _MetricsSinkSpy()

    svc = PipelineService(
        ingestion=ingestion,
        extractor=extractor,
        loader=loader,
        normalizer=normalizer,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
        metrics=metrics,
        embedding_model="text-embedding-3-small",
        chunker_version="v1",
        max_retries=1,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )

    await svc.run(ctx=ctx, doc_id=doc_id)

    # Dos intentos: 2 start_attempt y 2 end_attempt (falso y luego true)
    assert len([c for c in ingestion.calls if c[0] == "start_attempt"]) == 2
    ends = [c for c in ingestion.calls if c[0] == "end_attempt"]
    assert len(ends) == 2
    assert ends[0][-1] is False
    assert ends[1][-1] is True

    assert any(c[0] == "mark_ready" for c in ingestion.calls)
    assert not any(c[0] == "mark_failed" for c in ingestion.calls)

    stages = [s for s, _ in metrics.records]
    assert "attempt_error" in stages
    assert "attempt" in stages


@pytest.mark.asyncio
async def test_pipeline_failure_marks_failed_records_failed_and_sends_dlq_after_retries():
    class _BadExtractor:
        async def extract(self, *, tenant_id: str, doc_id):
            raise RuntimeError("boom")

    ctx = TenantContext(tenant_id="t1", user_id=uuid4(), role="user", scopes=[])
    doc_id = uuid4()

    ingestion = _IngestionSpy()
    extractor = _BadExtractor()
    loader = _Loader()
    normalizer = _Normalizer()
    chunker = _Chunker()
    embeddings = _Embeddings()
    vector_store = _VectorStoreSpy()
    metrics = _MetricsSinkSpy()
    dlq = _DLQSpy()

    svc = PipelineService(
        ingestion=ingestion,
        extractor=extractor,
        loader=loader,
        normalizer=normalizer,
        chunker=chunker,
        embeddings=embeddings,
        vector_store=vector_store,
        metrics=metrics,
        embedding_model="text-embedding-3-small",
        chunker_version="v1",
        max_retries=1,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        dlq=dlq,
    )

    with pytest.raises(RuntimeError):
        await svc.run(ctx=ctx, doc_id=doc_id)

    # Debe marcar failed tras agotar reintentos
    assert any(c[0] == "mark_failed" for c in ingestion.calls)

    stages = [s for s, _ in metrics.records]
    assert "attempt_error" in stages
    assert "failed" in stages

    # DLQ debe recibir 1 mensaje
    assert len(dlq.sent) == 1
    msg = dlq.sent[0]
    assert msg["tenant_id"] == "t1"
    assert msg["doc_id"] == doc_id
    assert msg["reason"].startswith("pipeline_failed_after_retries:RuntimeError")
    assert "trace" in msg["payload"]
    assert "error" in msg["payload"]
