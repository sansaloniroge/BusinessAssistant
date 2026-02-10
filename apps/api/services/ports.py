from __future__ import annotations

from typing import Any, Protocol, Sequence, TYPE_CHECKING

from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.filters import MetaFilters

if TYPE_CHECKING:
    from uuid import UUID
    from .retrieval_service import RetrievedChunk


class VectorStore(Protocol):
    async def search_by_embedding(
        self,
        *,
        tenant_id: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: MetaFilters,
    ) -> list[RetrievedChunk]:
        ...

    async def upsert_chunks(self, *, tenant_id: str, chunks: Sequence[dict[str, Any]]) -> int:
        """Inserta/actualiza chunks. Devuelve nº de filas afectadas."""
        ...

    async def delete_by_doc_id(self, *, tenant_id: str, doc_id: str) -> int:
        """Borra todos los chunks de un doc. Devuelve nº de filas borradas."""
        ...

    async def delete_by_chunk_id(self, *, tenant_id: str, chunk_id: str) -> int:
        """Borra un chunk concreto por su chunk_id. Devuelve nº de filas borradas."""
        ...

    async def health(self) -> bool:
        ...


class EmbeddingService(Protocol):
    async def embed_query(self, *, text: str) -> list[float]:
        ...

    async def embed_chunks(self, *, texts: Sequence[str]) -> list[Sequence[float]]:
        """Embebe múltiples textos en paralelo/por lotes."""
        ...


class PermissionsService(Protocol):
    async def vector_filters_for(self, ctx: TenantContext) -> MetaFilters:
        ...

    async def can_access_doc(self, *, ctx: TenantContext, doc_id: "UUID") -> bool:
        ...


class Reranker(Protocol):
    async def rerank(self, *, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...


# === Pipeline de ingesta ===
class Extractor(Protocol):
    async def extract(self, *, tenant_id: str, doc_id: "UUID") -> bytes | str:
        """Obtiene el contenido bruto de la fuente (blob/pdf/url/etc.)."""
        ...


class Loader(Protocol):
    async def load(self, *, raw: bytes | str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Parsea y devuelve estructura normalizada mínima: { title, language?, sections?, text }."""
        ...


class Normalizer(Protocol):
    async def normalize(self, *, doc: dict[str, Any]) -> dict[str, Any]:
        """Limpia/estandariza textos, títulos, separadores, etc."""
        ...


class Chunker(Protocol):
    async def chunk(self, *, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Devuelve lista de chunks con al menos: { index, title, content, metadata }."""
        ...


class MetricsSink(Protocol):
    async def record(self, *, tenant_id: str, doc_id: "UUID", stage: str, metrics: dict[str, Any]) -> None:
        """Registra métricas por etapa: latencia_ms, num_items, errores, contadores, etc."""
        ...


class DeadLetterQueue(Protocol):
    async def send(self, *, tenant_id: str, doc_id: "UUID", reason: str, payload: dict[str, Any]) -> None:
        """Envía a DLQ con motivo y payload para posterior reintento/análisis."""
        ...
