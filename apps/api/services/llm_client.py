from dataclasses import dataclass

from packages.shared.schemas.common import LLMUsage


@dataclass(slots=True, frozen=True)
class LLMResult:
    text: str
    usage: LLMUsage | None = None


class LLMClient:
    """Provider abstraction (OpenAI / Azure OpenAI / etc)."""

    async def generate(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        raise NotImplementedError
