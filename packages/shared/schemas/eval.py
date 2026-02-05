from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: str
    question: str
    expected_doc_ids: list[UUID] = Field(default_factory=list)
    notes: str | None = None
    created_at: datetime


class EvalRunRequest(BaseModel):
    tenant_id: str
    mode: str = "strict"
    model: str
    max_cases: int = Field(default=50, ge=1, le=500)


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    eval_case_id: UUID
    run_id: UUID
    overall: int = Field(ge=0, le=5)
    faithfulness: int = Field(ge=0, le=5)
    relevance: int = Field(ge=0, le=5)
    citation_quality: int = Field(ge=0, le=5)
    rationale: str


class EvalRunResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: str
    model: str
    created_at: datetime
    results: list[EvalCaseResult]
