from __future__ import annotations

import time
from typing import Any, Sequence
from uuid import UUID, uuid4

from packages.shared.schemas.common import TenantContext

from .ingestion_service import IngestionService
from .ports import Chunker, EmbeddingService, Extractor, Loader, MetricsSink, VectorStore


class PipelineService:
    """Pipeline de ingesta:
    extractor/loader → normalización → chunking → embeddings → upsert vector store → ready.
    Registra métricas por etapa y marca failed si ocurre error.
    """

    def __init__(
        self,
        *,
        ingestion: IngestionService,
        extractor: Extractor,
        loader: Loader,
        normalizer,
        chunker: Chunker,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        metrics: MetricsSink,
        embedding_model: str,
        chunker_version: str,
    ) -> None:
        self._ingestion = ingestion
        self._extractor = extractor
        self._loader = loader
        self._normalizer = normalizer
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._metrics = metrics
        self._embedding_model = embedding_model
        self._chunker_version = chunker_version

    async def run(self, *, ctx: TenantContext, doc_id: UUID, base_metadata: dict[str, Any] | None = None) -> None:
        base_metadata = dict(base_metadata or {})
        attempt_id = uuid4()

        await self._ingestion.start_processing(ctx=ctx, doc_id=doc_id)
        await self._ingestion.start_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id)

        try:
            # 1) Extract
            t0 = time.perf_counter()
            raw = await self._extractor.extract(tenant_id=str(ctx.tenant_id), doc_id=doc_id)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="extract", metrics={"latency_ms": self._ms(t0)}
            )

            # 2) Load/parse
            t0 = time.perf_counter()
            parsed = await self._loader.load(raw=raw, metadata=base_metadata)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="load", metrics={"latency_ms": self._ms(t0)}
            )

            # 3) Normalize
            t0 = time.perf_counter()
            doc = await self._normalizer.normalize(doc=parsed)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="normalize", metrics={"latency_ms": self._ms(t0)}
            )

            # 4) Chunking
            t0 = time.perf_counter()
            raw_chunks = await self._chunker.chunk(doc=doc)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="chunk",
                metrics={"latency_ms": self._ms(t0), "num_items": len(raw_chunks)},
            )

            # 5) Embeddings (batch)
            texts: Sequence[str] = [str(c.get("content", "")) for c in raw_chunks]
            t0 = time.perf_counter()
            vectors = await self._embeddings.embed_chunks(texts=texts)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="embed",
                metrics={"latency_ms": self._ms(t0), "num_items": len(vectors)},
            )

            if len(vectors) != len(raw_chunks):
                raise RuntimeError("embed_chunks devolvió un nº distinto de vectores que chunks")

            # 6) Upsert VectorStore
            t0 = time.perf_counter()
            to_upsert = []
            for i, c in enumerate(raw_chunks):
                chunk_id = self._ingestion.stable_chunk_id(doc_id=doc_id, index=int(c.get("index", i)))
                md = dict(base_metadata)
                md.update(dict(c.get("metadata") or {}))
                to_upsert.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "title": str(c.get("title", doc.get("title", "Untitled"))),
                        "content": str(c.get("content", "")),
                        "embedding": list(vectors[i]),
                        "department": md.get("department"),
                        "doc_type": md.get("doc_type"),
                        "tags": md.get("tags", []),
                        "doc_date": md.get("doc_date"),
                        "metadata": md,
                        "embedding_model": self._embedding_model,
                        "chunker_version": self._chunker_version,
                    }
                )

            affected = await self._vector_store.upsert_chunks(tenant_id=str(ctx.tenant_id), chunks=to_upsert)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="upsert",
                metrics={"latency_ms": self._ms(t0), "num_items": affected},
            )

            # 7) Mark ready
            await self._ingestion.mark_ready(ctx=ctx, doc_id=doc_id)
            await self._ingestion.end_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id, success=True)

        except Exception as e:
            # Mark failed and record error metric
            await self._ingestion.mark_failed(ctx=ctx, doc_id=doc_id)
            await self._ingestion.end_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id, success=False)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="error",
                metrics={"error": type(e).__name__, "message": str(e)},
            )
            # Re-raise to let caller handle
            raise

    @staticmethod
    def _ms(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)
