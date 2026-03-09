from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import asyncpg


@dataclass(slots=True, frozen=True)
class PersistedEvalRun:
    tenant_id: str
    eval_run_id: UUID


class PostgresEvalRunsRepo:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert_eval_run(
        self,
        *,
        tenant_id: str,
        eval_run_id: UUID,
        model: str,
        mode: str,
        max_cases: int,
    ) -> PersistedEvalRun:
        sql = """
        INSERT INTO eval_runs (tenant_id, eval_run_id, model, mode, max_cases)
        VALUES ($1, $2::uuid, $3, $4, $5)
        ON CONFLICT (tenant_id, eval_run_id) DO UPDATE SET
          model = EXCLUDED.model,
          mode = EXCLUDED.mode,
          max_cases = EXCLUDED.max_cases
        """

        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant_id))
            await conn.execute(sql, str(tenant_id), str(eval_run_id), str(model), str(mode), int(max_cases))

        return PersistedEvalRun(tenant_id=str(tenant_id), eval_run_id=eval_run_id)

