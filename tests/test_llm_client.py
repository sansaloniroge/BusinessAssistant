import pytest

from apps.api.services.llm_client import LLMClient, LLMResult
from packages.shared.schemas.common import LLMUsage


class _FakeLLM(LLMClient):
    def __init__(self, *, result: LLMResult | None = None, sleep_s: float = 0.0):
        self._result = result
        self._sleep_s = sleep_s

    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        if self._sleep_s:
            import asyncio

            await asyncio.sleep(self._sleep_s)
        assert self._result is not None
        return self._result


@pytest.mark.asyncio
async def test_generate_applies_timeout():
    llm = _FakeLLM(result=LLMResult(text="ok"), sleep_s=0.05)

    with pytest.raises(TimeoutError):
        await llm.generate(system="s", user="u", context="c", model="m", timeout_s=0.001)


@pytest.mark.asyncio
async def test_generate_normalizes_usage_when_none():
    llm = _FakeLLM(result=LLMResult(text="ok", usage=None))

    res = await llm.generate(system="s", user="u", context="c", model="m", timeout_s=1.0)

    assert res.usage is not None
    assert isinstance(res.usage, LLMUsage)
    assert res.usage.model == "m"
    assert res.usage.prompt_tokens == 0
    assert res.usage.completion_tokens == 0
    assert res.usage.total_tokens == 0
    assert res.usage.latency_ms >= 0


@pytest.mark.asyncio
async def test_generate_normalizes_usage_from_dict_payload():
    # En runtime, providers pueden devolver un dict/obj con shape diferente.
    llm = _FakeLLM(result=LLMResult(text="ok", usage=None))
    llm._result = LLMResult(text="ok", usage={"input_tokens": 3, "output_tokens": 5})  # type: ignore[arg-type]

    res = await llm.generate(system="s", user="u", context="c", model="m", timeout_s=1.0)

    assert res.usage is not None
    assert res.usage.prompt_tokens == 3
    assert res.usage.completion_tokens == 5
    assert res.usage.total_tokens == 8
