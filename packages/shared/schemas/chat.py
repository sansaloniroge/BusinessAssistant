from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from packages.shared.schemas.common import Citation, ConfidenceLevel, LLMUsage


class ChatMode(str, Enum):
    normal = "normal"
    strict = "strict"


class ChatFilters(BaseModel):
    department: str | None = None
    doc_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    date_from: datetime | None = None
    date_to: datetime | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    conversation_id: UUID | None = None
    mode: ChatMode = ChatMode.normal
    filters: ChatFilters | None = None

    # Debug/controls (OK to keep; hide in UI if you want)
    top_k: int = Field(default=12, ge=1, le=50)
    use_rerank: bool = True


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: UUID
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel
    follow_ups: list[str] = Field(default_factory=list)

    run_id: UUID
    usage: LLMUsage | None = None
    retrieval_debug: dict[str, object] | None = None
