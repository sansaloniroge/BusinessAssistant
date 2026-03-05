from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from apps.api.app import create_app
from apps.api.deps import get_chat_service, get_ctx, get_eval_service
from packages.shared.schemas.common import TenantContext


class _RunsRepo:
    def __init__(self):
        self.runs: dict[tuple[str, str], dict] = {}

    async def insert_run(self, *, run_id, data) -> None:
        self.runs[(str(data.tenant_id), str(run_id))] = {
            "tenant_id": str(data.tenant_id),
            "run_id": run_id,
            "question": data.question,
            "answer": data.answer,
            "retrieved_doc_ids": list(getattr(data, "retrieved_doc_ids", []) or []),
            "retrieval_debug": dict(getattr(data, "retrieval_debug", {}) or {}),
        }

    async def get_run(self, *, tenant_id: str, run_id):
        return self.runs.get((str(tenant_id), str(run_id)))


class _EvalRepo:
    def __init__(self):
        self.results = []

    async def insert_eval_result(self, **kwargs) -> None:
        self.results.append(kwargs)


class _ConversationsRepo:
    async def create(self, conversation_id, ctx) -> None:
        return None


class _MessagesRepo:
    async def insert(self, conversation_id, role, content) -> None:
        return None


@pytest.mark.asyncio
async def test_eval_smoke_end_to_end_without_postgres():
    app = create_app()

    tenant_id = str(uuid4())
    user_id = uuid4()

    runs_repo = _RunsRepo()
    eval_repo = _EvalRepo()

    async def _ctx_override() -> TenantContext:
        return TenantContext(tenant_id=tenant_id, user_id=user_id, role="user", scopes=[])

    # Override chat service para que use repos en memoria pero el resto del stack real.
    from apps.api.services.chat_service import ChatService
    from apps.api.services.citation_service import CitationService
    from apps.api.services.permissions import DefaultPermissionsService
    from apps.api.services.prompt_service import PromptService
    from apps.api.services.retrieval_service import RetrievalService
    from apps.api.deps import _DummyEmbeddings, _DummyLLM

    class _VS:
        async def search_by_embedding(self, *, tenant_id, query_embedding, top_k, filters):
            return []

    retrieval = RetrievalService(
        vector_store=_VS(),
        permissions=DefaultPermissionsService(),
        embeddings=_DummyEmbeddings(),
    )

    from apps.api.services.run_logger import RunLogger

    chat_svc = ChatService(
        retrieval=retrieval,
        prompts=PromptService(),
        llm=_DummyLLM(),
        citations=CitationService(),
        run_logger=RunLogger(runs_repo),
        conversations_repo=_ConversationsRepo(),
        messages_repo=_MessagesRepo(),
    )

    # Eval service real pero con repos en memoria.
    from apps.api.services.eval_judge_service import EvalJudgeService
    from apps.api.services.eval_service import EvalService

    eval_svc = EvalService(runs_repo=runs_repo, eval_repo=eval_repo, judge=EvalJudgeService(llm=_DummyLLM()))

    app.dependency_overrides[get_ctx] = _ctx_override

    async def _chat_override():
        return chat_svc

    async def _eval_override():
        return eval_svc

    app.dependency_overrides[get_chat_service] = _chat_override
    app.dependency_overrides[get_eval_service] = _eval_override

    client = TestClient(app)

    # 1) Ejecuta /chat → debe devolver run_id
    r = client.post("/chat", json={"message": "hola", "mode": "strict", "top_k": 5, "use_rerank": False})
    assert r.status_code == 200
    run_id = r.json().get("run_id")
    assert run_id

    # 2) Ejecuta /eval/.../judge → debe persistir en eval_repo
    j = client.post(f"/eval/runs/{run_id}/judge", json={})
    assert j.status_code == 200
    assert len(eval_repo.results) == 1

