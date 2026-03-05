from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from packages.shared.schemas.common import TenantContext
from packages.shared.schemas.eval import JudgeOutput

from apps.api.deps import get_ctx, get_eval_service
from apps.api.services.eval_service import EvalService

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/runs/{run_id}/judge", response_model=JudgeOutput)
async def judge_run(
    run_id: UUID,
    eval_run_id: UUID | None = Query(default=None),
    eval_case_id: UUID | None = Query(default=None),
    ctx: TenantContext = Depends(get_ctx),
    svc: EvalService = Depends(get_eval_service),
) -> JudgeOutput:
    try:
        return await svc.judge_existing_run(
            ctx=ctx,
            run_id=run_id,
            eval_run_id=eval_run_id,
            eval_case_id=eval_case_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found")
