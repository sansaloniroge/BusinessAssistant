import asyncio
from types import SimpleNamespace

import pytest

from apps.api.adapters.openai_llm_client import OpenAILLMClient


class _FakeCompletionsAPI:
    def __init__(self, *, text: str = "respuesta [C1]", sleep_s: float = 0.0) -> None:
        self._text = text
        self._sleep_s = sleep_s
        self.calls: list[dict] = []

    async def create(self, *, model: str, messages: list[dict]):
        self.calls.append({"model": model, "messages": messages})
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        message = SimpleNamespace(content=self._text)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAIClient:
    def __init__(self, *, text: str = "respuesta [C1]", sleep_s: float = 0.0) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletionsAPI(text=text, sleep_s=sleep_s))


@pytest.mark.asyncio
async def test_generate_returns_text_and_usage():
    client = _FakeOpenAIClient(text="hola [C1]")
    llm = OpenAILLMClient(client=client)

    res = await llm.generate(system="sys", user="pregunta", context="[C1] contenido", model="gpt-4.1-mini")

    assert res.text == "hola [C1]"
    assert res.usage is not None
    assert res.usage.prompt_tokens == 10
    assert res.usage.completion_tokens == 5
    assert res.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_generate_builds_messages_with_system_and_context():
    client = _FakeOpenAIClient()
    llm = OpenAILLMClient(client=client)

    await llm.generate(system="reglas del sistema", user="pregunta", context="[C1] contexto", model="gpt-4.1-mini")

    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["messages"][0] == {"role": "system", "content": "reglas del sistema"}
    assert "[C1] contexto" in call["messages"][1]["content"]
    assert "pregunta" in call["messages"][1]["content"]


@pytest.mark.asyncio
async def test_generate_without_context_uses_user_message_directly():
    client = _FakeOpenAIClient()
    llm = OpenAILLMClient(client=client)

    await llm.generate(system="sys", user="pregunta sola", context="", model="gpt-4.1-mini")

    call = client.chat.completions.calls[0]
    assert call["messages"][1] == {"role": "user", "content": "pregunta sola"}


@pytest.mark.asyncio
async def test_generate_applies_base_class_timeout():
    client = _FakeOpenAIClient(sleep_s=0.05)
    llm = OpenAILLMClient(client=client)

    with pytest.raises(TimeoutError):
        await llm.generate(system="sys", user="u", context="c", model="gpt-4.1-mini", timeout_s=0.001)


def test_missing_api_key_without_client_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAILLMClient()
