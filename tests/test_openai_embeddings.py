import asyncio
from types import SimpleNamespace

import pytest

from apps.api.adapters.openai_embeddings import OpenAIEmbeddingService


class _FakeEmbeddingsAPI:
    def __init__(self, *, dim: int = 1536, sleep_s: float = 0.0) -> None:
        self._dim = dim
        self._sleep_s = sleep_s
        self.calls: list[dict] = []

    async def create(self, *, model: str, input: list[str]):
        self.calls.append({"model": model, "input": list(input)})
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        data = [SimpleNamespace(embedding=[0.1] * self._dim) for _ in input]
        return SimpleNamespace(data=data)


class _FakeOpenAIClient:
    def __init__(self, *, dim: int = 1536, sleep_s: float = 0.0) -> None:
        self.embeddings = _FakeEmbeddingsAPI(dim=dim, sleep_s=sleep_s)


@pytest.mark.asyncio
async def test_embed_query_returns_vector_of_expected_dimension():
    client = _FakeOpenAIClient()
    svc = OpenAIEmbeddingService(client=client, model="text-embedding-3-small")

    vec = await svc.embed_query(text="hola")

    assert len(vec) == OpenAIEmbeddingService.EMBEDDING_DIM
    assert client.embeddings.calls == [{"model": "text-embedding-3-small", "input": ["hola"]}]


@pytest.mark.asyncio
async def test_embed_chunks_batches_in_a_single_call():
    client = _FakeOpenAIClient()
    svc = OpenAIEmbeddingService(client=client, model="text-embedding-3-small")

    texts = ["a", "b", "c"]
    vectors = await svc.embed_chunks(texts=texts)

    assert len(vectors) == 3
    assert all(len(v) == OpenAIEmbeddingService.EMBEDDING_DIM for v in vectors)
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == texts


@pytest.mark.asyncio
async def test_embed_chunks_empty_does_not_call_api():
    client = _FakeOpenAIClient()
    svc = OpenAIEmbeddingService(client=client)

    vectors = await svc.embed_chunks(texts=[])

    assert vectors == []
    assert client.embeddings.calls == []


@pytest.mark.asyncio
async def test_dimension_mismatch_raises_value_error():
    client = _FakeOpenAIClient(dim=1024)
    svc = OpenAIEmbeddingService(client=client)

    with pytest.raises(ValueError, match="dimensión"):
        await svc.embed_query(text="hola")


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error():
    client = _FakeOpenAIClient(sleep_s=0.05)
    svc = OpenAIEmbeddingService(client=client, timeout_s=0.001)

    with pytest.raises(TimeoutError):
        await svc.embed_query(text="hola")


def test_missing_api_key_without_client_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingService()
