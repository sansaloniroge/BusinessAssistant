from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from apps.api.services.ports import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    """Embeddings provider real (OpenAI).

    Modelo por defecto: text-embedding-3-small (1536 dims), que coincide con el
    VECTOR(1536) ya fijado en el esquema (documents_chunks) — activar este
    provider no requiere migración.
    """

    EMBEDDING_DIM: int = 1536

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise ValueError("OPENAI_API_KEY no configurado")
        self._client = client if client is not None else AsyncOpenAI(api_key=api_key)
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._timeout_s = float(timeout_s)

    async def embed_query(self, *, text: str) -> list[float]:
        vectors = await self._embed([text])
        return vectors[0]

    async def embed_chunks(self, *, texts: Sequence[str]) -> list[Sequence[float]]:
        if not texts:
            return []
        return await self._embed(list(texts))

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            res = await asyncio.wait_for(
                self._client.embeddings.create(model=self._model, input=inputs),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"OpenAI embeddings timeout after {self._timeout_s:.1f}s") from e

        vectors = [list(item.embedding) for item in res.data]

        for v in vectors:
            if len(v) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"OpenAI devolvió embeddings de dimensión {len(v)}, esperado {self.EMBEDDING_DIM} "
                    f"(modelo={self._model})"
                )

        return vectors
