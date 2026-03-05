from __future__ import annotations

from uuid import UUID

from packages.shared.schemas.eval_dataset import EvalCaseFixture


def fixtures_for_tenant(*, tenant_id: str, user_id: UUID) -> list[EvalCaseFixture]:
    """Fixtures mínimas por tenant.

    En esta fase, son preguntas smoke/regresión. En un proyecto real, esto vendría
    versionado por vertical (HR/IT/Security) y por modo (strict/normal).
    """

    return [
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Cuál es la política de vacaciones?",
            mode="strict",
            notes="Smoke: pregunta de política genérica; debería rechazar si no hay evidencia.",
            tags=["smoke", "hr"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="Resume el proceso de onboarding en 3 pasos.",
            mode="strict",
            notes="Smoke: proceso; en vacío debería rechazar.",
            tags=["smoke", "process"],
        ),
    ]

