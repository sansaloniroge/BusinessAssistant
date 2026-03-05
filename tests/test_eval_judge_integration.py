import json
from uuid import uuid4

import pytest

from apps.api.services.eval_judge_service import EvalJudgeService
from apps.api.services.eval_service import EvalService
from apps.api.services.llm_client import LLMClient, LLMResult
from packages.shared.schemas.eval import JudgeOutput


class _FixedJSONLLM(LLMClient):
    def __init__(self, payload: dict):
        super().__init__()
        self._payload = payload

    async def _generate_impl(self, *, system: str, user: str, context: str, model: str) -> LLMResult:
        # Devuelve JSON válido como texto. Usage dummy.
        return LLMResult(text=json.dumps(self._payload), usage={"total_tokens": 10})


class _RunsRepo:
    def __init__(self, run_row: dict | None):
        self._run_row = run_row
        self.calls = []

    async def get_run(self, *, tenant_id: str, run_id):
        self.calls.append((tenant_id, run_id))
        return self._run_row


class _EvalRepo:
    def __init__(self):
        self.inserted = []

    async def insert_eval_result(
        self,
        *,
        tenant_id,
        eval_result_id,
        eval_run_id,
        eval_case_id,
        run_id,
        judge_model,
        judge_usage,
        output,
    ):
        self.inserted.append(
            {
                "tenant_id": tenant_id,
                "eval_result_id": eval_result_id,
                "eval_run_id": eval_run_id,
                "eval_case_id": eval_case_id,
                "run_id": run_id,
                "judge_model": judge_model,
                "judge_usage": judge_usage,
                "output": output,
            }
        )


@pytest.mark.asyncio
async def test_eval_service_judges_real_run_and_persists_result(tenant_ctx):
    run_id = uuid4()

    run_row = {
        "tenant_id": str(tenant_ctx.tenant_id),
        "run_id": run_id,
        "question": "Q?",
        "answer": "A [C1]",
        "retrieved_doc_ids": ["d1"],
        "retrieval_debug": {"scores": [0.9], "used_chunk_ids": ["c1"], "used_doc_ids": ["d1"]},
    }

    expected = {
        "overall": 4,
        "faithfulness": 5,
        "relevance": 4,
        "citation_quality": 4,
        "refusal_correctness": 5,
        "rationale": "ok",
    }

    llm = _FixedJSONLLM(expected)
    judge = EvalJudgeService(llm=llm)

    runs_repo = _RunsRepo(run_row)
    eval_repo = _EvalRepo()

    svc = EvalService(runs_repo=runs_repo, eval_repo=eval_repo, judge=judge)

    out = await svc.judge_existing_run(ctx=tenant_ctx, run_id=run_id)

    assert isinstance(out, JudgeOutput)
    assert out.overall == 4
    assert out.faithfulness == 5
    assert out.refusal_correctness == 5

    # se persiste 1 resultado asociado al run
    assert len(eval_repo.inserted) == 1
    inserted = eval_repo.inserted[0]
    assert inserted["tenant_id"] == str(tenant_ctx.tenant_id)
    assert inserted["run_id"] == run_id
    assert isinstance(inserted["output"], JudgeOutput)


@pytest.mark.asyncio
async def test_eval_service_returns_404_like_when_run_missing(tenant_ctx):
    run_id = uuid4()

    llm = _FixedJSONLLM(
        {
            "overall": 0,
            "faithfulness": 0,
            "relevance": 0,
            "citation_quality": 0,
            "refusal_correctness": 0,
            "rationale": "n/a",
        }
    )
    judge = EvalJudgeService(llm=llm)

    svc = EvalService(runs_repo=_RunsRepo(None), eval_repo=_EvalRepo(), judge=judge)

    with pytest.raises(KeyError):
        await svc.judge_existing_run(ctx=tenant_ctx, run_id=run_id)

