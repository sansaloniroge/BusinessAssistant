from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvalCaseFixture(BaseModel):
    """Caso de evaluación (fixture) para ejecutar contra la API real.

    Nota: esto es un artefacto de test/runner (no necesariamente parte del contrato público HTTP).
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    user_id: UUID
    conversation_id: UUID | None = None

    question: str
    mode: str = "strict"

    # Opcional: qué docs esperas que aparezcan en retrieval (si aplica).
    expected_doc_ids: list[UUID] = Field(default_factory=list)

    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvalCaseArtifact(BaseModel):
    """Artefacto capturado por el runner tras ejecutar un caso."""

    model_config = ConfigDict(from_attributes=True)

    eval_case_id: str
    tenant_id: str
    created_at: datetime

    request: dict
    response: dict
    run_id: UUID

    latency_ms: int | None = None
    usage: dict | None = None

