from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from packages.shared.schemas.common import LLMUsage


@dataclass(slots=True, frozen=True)
class RunLogInput:
    tenant_id: UUID
    user_id: UUID
    conversation_id: UUID
    question: str
    answer: str
    model: str
    usage: LLMUsage | None
    retrieved_doc_ids: list[str]
    retrieval_debug: dict[str, Any]


class RunLogger:
    def __init__(self, runs_repo) -> None:
        self._runs_repo = runs_repo

    async def persist(self, data: RunLogInput) -> UUID:
        run_id = uuid4()
        await self._runs_repo.insert_run(run_id=run_id, data=data)
        return run_id
