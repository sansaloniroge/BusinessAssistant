from __future__ import annotations

from uuid import UUID

from packages.shared.schemas.eval_dataset import EvalCaseFixture


def fixtures_for_tenant(*, tenant_id: str, user_id: UUID) -> list[EvalCaseFixture]:
    """Fixtures por tenant, alineadas con los 5 documentos sintéticos ingestados
    por `scripts/ingest_documents.py` (ver `scripts/sample_docs/`) para el tenant
    `tenant_test`.

    Incluye una pregunta con evidencia real por documento (debe responder con
    citas) y varias fuera de dominio (deben rechazar por falta de evidencia,
    ejercitando `refusal_correctness`). En un proyecto real esto vendría
    versionado por vertical (HR/IT/Finance) y por modo (strict/normal).
    """

    return [
        # Con evidencia real (una por documento ingestado)
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="Si trabajo en remoto, ¿cuántos días al año puedo trabajar desde otro país sin avisar a nadie?",
            mode="strict",
            notes="Con evidencia: remote_work_policy.md especifica 60 días/año con aviso previo a RRHH.",
            tags=["grounded", "hr", "remote_work_policy"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Qué debo hacer durante mi primera semana como nueva incorporación?",
            mode="strict",
            notes="Con evidencia: onboarding_guide.md (formación obligatoria, 1:1s con el equipo).",
            tags=["grounded", "hr", "onboarding_guide"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Es obligatorio el doble factor de autenticación para acceder al correo corporativo?",
            mode="strict",
            notes="Con evidencia: it_security_policy.md (MFA obligatoria, no desactivable).",
            tags=["grounded", "it", "it_security_policy"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Cuál es el límite de dieta diaria en un viaje de trabajo dentro de España?",
            mode="strict",
            notes="Con evidencia: expense_reimbursement.md (45 EUR/día en España).",
            tags=["grounded", "finance", "expense_reimbursement"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Con qué frecuencia hay que publicar actualizaciones durante un incidente Sev1 abierto?",
            mode="strict",
            notes="Con evidencia: incident_response_runbook.md (cada 20 minutos).",
            tags=["grounded", "it", "incident_response_runbook"],
        ),
        # Fuera de dominio: no hay ningún documento ingestado sobre esto, debería rechazar
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Cuál es la política de vacaciones y días festivos de la empresa?",
            mode="strict",
            notes="Sin evidencia: ningún documento ingestado cubre vacaciones/festivos; debe rechazar.",
            tags=["refusal", "hr"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Cuál es el stack tecnológico que usa el equipo de backend?",
            mode="strict",
            notes="Sin evidencia: no hay documentación técnica ingestada; debe rechazar.",
            tags=["refusal", "engineering"],
        ),
        EvalCaseFixture(
            tenant_id=tenant_id,
            user_id=user_id,
            question="¿Cuánto cuesta el seguro médico privado que ofrece la empresa?",
            mode="strict",
            notes="Sin evidencia: no hay documentación de beneficios/seguro ingestada; debe rechazar.",
            tags=["refusal", "hr"],
        ),
    ]

