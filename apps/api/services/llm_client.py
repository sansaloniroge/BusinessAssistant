from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from packages.shared.schemas.common import LLMUsage


@dataclass(slots=True, frozen=True)
class LLMResult:
    text: str
    usage: LLMUsage | Any | None = None
    latency_ms: int | None = None


def normalize_usage(*, model: str, raw: Any, latency_ms: int) -> LLMUsage:
    """Normaliza usage de providers.

    Acepta dicts/objetos/lib-specific usages y devuelve un LLMUsage consistente.
    """

    def get(obj: Any, *keys: str, default: Any = 0) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    return obj[k]
            return default
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)
        return default

    prompt = int(get(raw, "prompt_tokens", "input_tokens", default=0) or 0)
    completion = int(get(raw, "completion_tokens", "output_tokens", default=0) or 0)
    total = int(get(raw, "total_tokens", default=(prompt + completion)) or 0)
    cost = float(get(raw, "cost_estimate_usd", "cost", default=0.0) or 0.0)

    return LLMUsage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cost_estimate_usd=cost,
        latency_ms=latency_ms,
    )


class LLMClient:
    """Provider abstraction (OpenAI / Azure OpenAI / etc)."""

    default_timeout_s: float = 45.0

    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        raise NotImplementedError

    async def generate(
        self,
        *,
        system: str,
        user: str,
        context: str,
        model: str,
        timeout_s: float | None = None,
    ) -> LLMResult:
        t0 = time.perf_counter()
        to = self.default_timeout_s if timeout_s is None else float(timeout_s)

        try:
            res = await asyncio.wait_for(
                self._generate_impl(system=system, user=user, context=context, model=model),
                timeout=to,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"LLM generate timeout after {to:.1f}s") from e

        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage_raw = getattr(res, "usage", None)

        # Normaliza si:
        # - viene None
        # - viene en formato dict/obj
        # - o viene como LLMUsage sin latencia informada
        if not isinstance(usage_raw, LLMUsage) or int(getattr(usage_raw, "latency_ms", 0) or 0) == 0:
            usage = normalize_usage(model=model, raw=usage_raw, latency_ms=latency_ms)
        else:
            usage = usage_raw

        return LLMResult(text=res.text, usage=usage, latency_ms=latency_ms)
