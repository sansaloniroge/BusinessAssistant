from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from apps.api.services.llm_client import LLMClient, LLMResult


class OpenAILLMClient(LLMClient):
    """Provider real (OpenAI chat completions).

    El timeout se gestiona en la clase base (`LLMClient.generate`), que envuelve
    `_generate_impl` en `asyncio.wait_for` — este adaptador no necesita su propio
    manejo de timeout.
    """

    def __init__(self, *, client: Any | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise ValueError("OPENAI_API_KEY no configurado")
        self._client = client if client is not None else AsyncOpenAI(api_key=api_key)

    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        user_content = f"Context:\n{context}\n\nQuestion:\n{user}" if context else user

        res = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )

        text = res.choices[0].message.content or ""
        return LLMResult(text=text, usage=res.usage)
