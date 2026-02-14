from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer


@dataclass(slots=True, frozen=True)
class ObservabilityConfig:
    service_name: str = "businessassistant"
    environment: str = "dev"
    otlp_endpoint: str = ""
    enable_console_debug: bool = False

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", os.getenv("SERVICE_NAME", "businessassistant")),
            environment=os.getenv("ENV", os.getenv("OTEL_ENVIRONMENT", "dev")),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
            enable_console_debug=os.getenv("OBS_CONSOLE_DEBUG", "false").lower() in {"1", "true", "yes"},
        )


_INITIALIZED = False


def setup_observability(cfg: ObservabilityConfig | None = None) -> None:
    """Inicializa providers de trazas y métricas.

    - Si hay OTEL_EXPORTER_OTLP_ENDPOINT exporta por OTLP/gRPC.
    - Si no, queda en no-op (útil para tests/local sin collector).

    Idempotente.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    cfg = cfg or ObservabilityConfig.from_env()

    resource = Resource.create(
        {
            "service.name": cfg.service_name,
            "deployment.environment": cfg.environment,
        }
    )

    # Traces
    tracer_provider = TracerProvider(resource=resource)

    if cfg.otlp_endpoint:
        span_exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(tracer_provider)

    # Metrics
    readers = []
    if cfg.otlp_endpoint:
        metric_exporter = OTLPMetricExporter(endpoint=cfg.otlp_endpoint)
        readers.append(PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10_000))

    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))

    _INITIALIZED = True


def get_tracer(name: str = "apps.api") -> Tracer:
    return trace.get_tracer(name)


def get_meter(name: str = "apps.api"):
    return metrics.get_meter(name)


def _attrs_safe(d: dict[str, Any] | None) -> dict[str, Any]:
    """Evita atributos no-serializables / demasiado grandes."""
    if not d:
        return {}

    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            # Mantener listas pequeñas de tipos simples
            if len(v) <= 50 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = [x for x in v if x is not None]
            else:
                out[k] = f"<{type(v).__name__}:{len(v)}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out

