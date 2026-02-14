from __future__ import annotations

import time
from typing import Any, Sequence
from uuid import UUID, uuid4

from packages.shared.schemas.common import TenantContext

from .ingestion_service import IngestionService
from .observability import get_meter, get_tracer
from .ports import Chunker, EmbeddingService, Extractor, Loader, MetricsSink, VectorStore, DeadLetterQueue


class PipelineService:
    """Pipeline de ingesta:
    extractor/loader → normalización → chunking → embeddings → upsert vector store → ready.
    Fiabilidad: reintentos con backoff, DLQ, límites; métricas y trazas por intento.
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
        # Fiabilidad
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        dlq: DeadLetterQueue | None = None,
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
        self._max_retries = int(max_retries)
        self._backoff_base_s = float(backoff_base_s)
        self._backoff_max_s = float(backoff_max_s)
        self._dlq = dlq

        # OTel
        self._tracer = get_tracer("apps.api.pipeline")
        self._meter = get_meter("apps.api.pipeline")

        self._attempts_total = self._meter.create_counter(
            "ingestion_attempts_total",
            description="Intentos totales de ingesta",
        )
        self._attempt_errors_total = self._meter.create_counter(
            "ingestion_attempt_errors_total",
            description="Errores por intento de ingesta",
        )
        self._stage_latency_ms = self._meter.create_histogram(
            "ingestion_stage_latency_ms",
            unit="ms",
            description="Latencia por etapa del pipeline",
        )

    async def run(self, *, ctx: TenantContext, doc_id: UUID, base_metadata: dict[str, Any] | None = None) -> None:
        base_metadata = dict(base_metadata or {})

        attrs = {"tenant_id": str(ctx.tenant_id), "doc_id": str(doc_id)}

        with self._tracer.start_as_current_span("pipeline.run", attributes=attrs) as span:
            await self._ingestion.start_processing(ctx=ctx, doc_id=doc_id)

            attempt_num = 0
            last_error: Exception | None = None
            trace: dict[str, Any] = {"tenant_id": str(ctx.tenant_id), "doc_id": str(doc_id), "attempts": []}

            while attempt_num <= self._max_retries:
                attempt_num += 1
                attempt_id = uuid4()

                attempt_attrs = {**attrs, "attempt_num": int(attempt_num), "attempt_id": str(attempt_id)}
                self._attempts_total.add(1, attributes=attempt_attrs)

                await self._ingestion.start_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id)

                try:
                    with self._tracer.start_as_current_span("pipeline.attempt", attributes=attempt_attrs):
                        await self._run_once(
                            ctx=ctx,
                            doc_id=doc_id,
                            base_metadata=base_metadata,
                            attempt_id=attempt_id,
                            trace=trace,
                            attrs=attempt_attrs,
                        )

                    # éxito
                    await self._ingestion.mark_ready(ctx=ctx, doc_id=doc_id)
                    await self._ingestion.end_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id, success=True)
                    await self._metrics.record(
                        tenant_id=str(ctx.tenant_id),
                        doc_id=doc_id,
                        stage="attempt",
                        metrics={"attempt_num": attempt_num, "success": True},
                    )
                    span.set_attribute("pipeline.success", True)
                    span.set_attribute("pipeline.attempts", int(attempt_num))
                    return

                except Exception as e:
                    last_error = e
                    await self._ingestion.end_attempt(ctx=ctx, doc_id=doc_id, attempt_id=attempt_id, success=False)

                    self._attempt_errors_total.add(
                        1,
                        attributes={**attempt_attrs, "error.type": type(e).__name__},
                    )
                    span.record_exception(e)
                    span.set_attribute("pipeline.last_error", type(e).__name__)

                    await self._metrics.record(
                        tenant_id=str(ctx.tenant_id),
                        doc_id=doc_id,
                        stage="attempt_error",
                        metrics={
                            "attempt_num": attempt_num,
                            "error": type(e).__name__,
                            "message": str(e),
                            "error_rate": 1.0,
                        },
                    )

                    if attempt_num <= self._max_retries:
                        delay = min(self._backoff_max_s, self._backoff_base_s * (2 ** (attempt_num - 1)))
                        await self._metrics.record(
                            tenant_id=str(ctx.tenant_id),
                            doc_id=doc_id,
                            stage="backoff",
                            metrics={"attempt_num": attempt_num, "delay_s": round(delay, 3)},
                        )
                        span.set_attribute("pipeline.backoff_s", float(delay))

                        # Mantener comportamiento actual (busy-wait) pero registrar span.
                        t_sleep = time.perf_counter()
                        while (time.perf_counter() - t_sleep) < max(0.0, delay * 0.9):
                            pass
                    else:
                        break

            await self._ingestion.mark_failed(ctx=ctx, doc_id=doc_id)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="failed",
                metrics={"attempts": attempt_num, "reason": type(last_error).__name__ if last_error else "unknown"},
            )
            span.set_attribute("pipeline.success", False)

            if self._dlq and last_error is not None:
                await self._dlq.send(
                    tenant_id=str(ctx.tenant_id),
                    doc_id=doc_id,
                    reason=f"pipeline_failed_after_retries:{type(last_error).__name__}",
                    payload={"error": str(last_error), "trace": trace},
                )

            if last_error:
                raise last_error

    async def _run_once(
        self,
        *,
        ctx: TenantContext,
        doc_id: UUID,
        base_metadata: dict[str, Any],
        attempt_id: UUID,
        trace: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        attempt_trace: dict[str, Any] = {"attempt_id": str(attempt_id), "stages": []}

        def rec_stage(stage: str, t0: float, extra: dict[str, Any] | None = None) -> None:
            ms = self._ms(t0)
            attempt_trace["stages"].append({"stage": stage, "latency_ms": ms, **(extra or {})})
            self._stage_latency_ms.record(ms, attributes={**attrs, "stage": stage})

        # 1) Extract
        with self._tracer.start_as_current_span("pipeline.extract", attributes=attrs):
            t0 = time.perf_counter()
            raw = await self._extractor.extract(tenant_id=str(ctx.tenant_id), doc_id=doc_id)
            rec_stage("extract", t0)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="extract", metrics={"latency_ms": self._ms(t0)}
            )

        # 2) Load/parse
        with self._tracer.start_as_current_span("pipeline.load", attributes=attrs):
            t0 = time.perf_counter()
            parsed = await self._loader.load(raw=raw, metadata=base_metadata)
            rec_stage("load", t0)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="load", metrics={"latency_ms": self._ms(t0)}
            )

        # 3) Normalize
        with self._tracer.start_as_current_span("pipeline.normalize", attributes=attrs):
            t0 = time.perf_counter()
            doc = await self._normalizer.normalize(doc=parsed)
            rec_stage("normalize", t0)
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id), doc_id=doc_id, stage="normalize", metrics={"latency_ms": self._ms(t0)}
            )

        # 4) Chunking
        with self._tracer.start_as_current_span("pipeline.chunk", attributes=attrs):
            t0 = time.perf_counter()
            raw_chunks = await self._chunker.chunk(doc=doc)
            rec_stage("chunk", t0, {"num_items": len(raw_chunks)})
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="chunk",
                metrics={"latency_ms": self._ms(t0), "num_items": len(raw_chunks)},
            )

        # 5) Embeddings (batch)
        texts: Sequence[str] = [str(c.get("content", "")) for c in raw_chunks]
        with self._tracer.start_as_current_span("pipeline.embed", attributes={**attrs, "num_items": len(texts)}):
            t0 = time.perf_counter()
            vectors = await self._embeddings.embed_chunks(texts=texts)
            rec_stage("embed", t0, {"num_items": len(vectors)})
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="embed",
                metrics={"latency_ms": self._ms(t0), "num_items": len(vectors)},
            )

        if len(vectors) != len(raw_chunks):
            raise RuntimeError("embed_chunks devolvió un nº distinto de vectores que chunks")

        # 6) Upsert VectorStore
        with self._tracer.start_as_current_span("pipeline.upsert", attributes={**attrs, "num_items": len(raw_chunks)}):
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
            rec_stage("upsert", t0, {"num_items": affected})
            await self._metrics.record(
                tenant_id=str(ctx.tenant_id),
                doc_id=doc_id,
                stage="upsert",
                metrics={"latency_ms": self._ms(t0), "num_items": affected},
            )

        trace["attempts"].append(attempt_trace)

    @staticmethod
    def _ms(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)
