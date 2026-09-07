from __future__ import annotations

import os
from typing import Any

from openai import AsyncAzureOpenAI

from apps.api.services.llm_client import LLMClient, LLMResult


class AzureOpenAILLMClient(LLMClient):
    """Provider real (Azure OpenAI chat completions).

    Mismo patrón Adapter que `OpenAILLMClient`: ambos implementan `LLMClient`,
    así que `ChatService`/`EvalJudgeService` no saben (ni les importa) contra
    qué proveedor hablan.

    Diferencia clave con OpenAI directo: Azure enruta por *deployment name*
    (un alias que tú eliges al desplegar un modelo en tu recurso Azure), no
    por el nombre de familia del modelo (`gpt-4.1-mini`). Por simplicidad —
    suficiente para un portfolio, no para producción multi-modelo — este
    adaptador ignora el `model` que le pasa `LLMClient.generate()` y siempre
    usa el único deployment configurado en `AZURE_OPENAI_DEPLOYMENT`.

    El timeout se gestiona en la clase base (`LLMClient.generate`), igual que
    en `OpenAILLMClient`.
    """

    def __init__(self, *, client: Any | None = None, deployment: str | None = None) -> None:
        self._deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if client is None:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

            missing = [
                name
                for name, value in (
                    ("AZURE_OPENAI_ENDPOINT", endpoint),
                    ("AZURE_OPENAI_API_KEY", api_key),
                    ("AZURE_OPENAI_DEPLOYMENT", self._deployment),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"Faltan variables de entorno para Azure OpenAI: {', '.join(missing)}")

            client = AsyncAzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        elif not self._deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT no configurado")

        self._client = client

    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        user_content = f"Context:\n{context}\n\nQuestion:\n{user}" if context else user

        res = await self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )

        text = res.choices[0].message.content or ""
        return LLMResult(text=text, usage=res.usage)
