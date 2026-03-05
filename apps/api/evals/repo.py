from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import asyncpg

from packages.shared.schemas.eval_dataset import EvalCaseFixture


@dataclass(slots=True, frozen=True)
class PersistedEvalCase:
    tenant_id: str
    eval_case_id: UUID


class PostgresEvalCasesRepo:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_case(self, *, case_id: UUID, case: EvalCaseFixture) -> PersistedEvalCase:
        sql = """
        INSERT INTO eval_cases (tenant_id, eval_case_id, question, expected_doc_ids, notes)
        VALUES ($1, $2::uuid, $3, $4::jsonb, $5)
        ON CONFLICT (tenant_id, eval_case_id) DO UPDATE SET
          question = EXCLUDED.question,
          expected_doc_ids = EXCLUDED.expected_doc_ids,
          notes = EXCLUDED.notes
        """

        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(case.tenant_id))
            await conn.execute(
                sql,
                str(case.tenant_id),
                str(case_id),
                str(case.question),
                [str(x) for x in (case.expected_doc_ids or [])],
                str(case.notes) if case.notes is not None else None,
            )

        return PersistedEvalCase(tenant_id=str(case.tenant_id), eval_case_id=case_id)

