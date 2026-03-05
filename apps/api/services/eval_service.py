from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.eval import JudgeInput, JudgeOutput

from apps.api.services.eval_judge_service import EvalJudgeService


@dataclass(slots=True, frozen=True)
class EvalResultPersistInput:
    tenant_id: str
    eval_run_id: UUID | None
    eval_case_id: UUID | None
    run_id: UUID
    judge_model: str
    judge_usage: dict | None
    output: JudgeOutput


class EvalService:
    def __init__(
        self,
        *,
        runs_repo,
        eval_repo,
        judge: EvalJudgeService,
        default_mode: str = "strict",
    ) -> None:
        self._runs_repo = runs_repo
        self._eval_repo = eval_repo
        self._judge = judge
        self._default_mode = default_mode

    async def judge_existing_run(self, *, ctx: TenantContext, run_id: UUID) -> JudgeOutput:
        run = await self._runs_repo.get_run(tenant_id=str(ctx.tenant_id), run_id=run_id)
        if run is None:
            raise KeyError("run_not_found")

        inp = JudgeInput(
            tenant_id=str(ctx.tenant_id),
            run_id=run_id,
            question=str(run["question"]),
            answer=str(run["answer"]),
            citations=[],
            retrieved_doc_ids=list(run.get("retrieved_doc_ids") or []),
            retrieval_debug=dict(run.get("retrieval_debug") or {}),
            mode=str(run.get("retrieval_debug", {}).get("mode") or self._default_mode),
        )

        out, usage = await self._judge.judge(inp=inp)

        usage_json = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else None)

        await self._eval_repo.insert_eval_result(
            tenant_id=str(ctx.tenant_id),
            eval_result_id=uuid4(),
            eval_run_id=None,
            eval_case_id=None,
            run_id=run_id,
            judge_model=getattr(usage, "model", None) or "(unknown)",
            judge_usage=usage_json,
            output=out,
        )

        return out

