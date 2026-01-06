from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConfidenceLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TenantContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: UUID
    user_id: UUID
    role: str
    scopes: list[str] = Field(default_factory=list)


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate_usd: float = 0.0
    latency_ms: int = 0


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: UUID
    title: str
    chunk_id: str
    snippet: str

    section: str | None = None
    page: int | None = None

    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=utcnow)
