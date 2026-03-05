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

    @staticmethod
    def _maybe_uuid(v: Any) -> UUID | None:
        if v is None:
            return None
        if isinstance(v, UUID):
            return v
        try:
            return UUID(str(v))
        except Exception:
            return None

    async def judge_existing_run(
        self,
        *,
        ctx: TenantContext,
        run_id: UUID,
        eval_run_id: UUID | None = None,
        eval_case_id: UUID | None = None,
    ) -> JudgeOutput:
        run = await self._runs_repo.get_run(tenant_id=str(ctx.tenant_id), run_id=run_id)
        if run is None:
            raise KeyError("run_not_found")

        retrieval_debug = dict(run.get("retrieval_debug") or {})

        # Si no vienen explícitos, intentar inferir desde debug (si el runner los inyecta en el request y se loguean).
        inferred_eval_run_id = self._maybe_uuid(retrieval_debug.get("eval_run_id"))
        inferred_eval_case_id = self._maybe_uuid(retrieval_debug.get("eval_case_id"))

        eval_run_id_eff = eval_run_id or inferred_eval_run_id
        eval_case_id_eff = eval_case_id or inferred_eval_case_id

        inp = JudgeInput(
            tenant_id=str(ctx.tenant_id),
            run_id=run_id,
            question=str(run["question"]),
            answer=str(run["answer"]),
            citations=[],
            retrieved_doc_ids=list(run.get("retrieved_doc_ids") or []),
            retrieval_debug=retrieval_debug,
            mode=str(retrieval_debug.get("mode") or self._default_mode),
        )

        out, usage = await self._judge.judge(inp=inp)

        usage_json = (
            usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else (usage if isinstance(usage, dict) else None)
        )

        await self._eval_repo.insert_eval_result(
            tenant_id=str(ctx.tenant_id),
            eval_result_id=uuid4(),
            eval_run_id=eval_run_id_eff,
            eval_case_id=eval_case_id_eff,
            run_id=run_id,
            judge_model=getattr(usage, "model", None) or "(unknown)",
            judge_usage=usage_json,
            output=out,
        )

        return out
