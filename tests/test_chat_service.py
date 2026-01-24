import pytest
from uuid import uuid4

from apps.api.services.chat_service import ChatService
from apps.api.services.llm_client import LLMResult
from apps.api.services.retrieval_service import RetrievedChunk, RetrievalResult
from packages.shared.schemas.chat import ChatMode, ChatRequest
from packages.shared.schemas.common import ConfidenceLevel, Citation


class _Retrieval:
    def __init__(self, result: RetrievalResult):
        self._result = result

    async def retrieve(self, **kwargs):
        return self._result


class _Prompts:
    def system_prompt(self, mode):
        return "SYS"

    def build_context(self, chunks):
        return "CTX"


class _LLM:
    def __init__(self, text: str):
        self._text = text

    async def generate(self, *, system, user, context, model):
        return LLMResult(text=self._text, usage=None)


class _LLMTimeout:
    async def generate(self, *, system, user, context, model):
        raise TimeoutError("LLM generate timeout")


class _Citations:
    def __init__(self, citations, strict_ok: bool = True):
        self._citations = citations
        self._strict_ok = strict_ok

    def build_citations(self, chunks, answer):
        return list(self._citations)

    def validate_strict(self, answer, citations):
        return self._strict_ok


class _RunRepo:
    def __init__(self):
        self.inserted = []

    async def insert_run(self, *, run_id, data):
        self.inserted.append((run_id, data))


class _RunLogger:
    def __init__(self, repo):
        self._repo = repo

    async def persist(self, data):
        rid = uuid4()
        await self._repo.insert_run(run_id=rid, data=data)
        return rid


class _ConversationsRepo:
    def __init__(self):
        self.created = []

    async def create(self, conversation_id, ctx):
        self.created.append((conversation_id, ctx))


class _MessagesRepo:
    def __init__(self):
        self.messages = []

    async def insert(self, conversation_id, role, content):
        self.messages.append((conversation_id, role, content))


@pytest.mark.asyncio
async def test_chat_service_refuses_when_low_evidence(tenant_ctx):
    r = RetrievalResult(chunks=[], evidence_strength=0.0, debug={"x": 1})

    convo = _ConversationsRepo()
    msgs = _MessagesRepo()
    run_repo = _RunRepo()

    svc = ChatService(
        retrieval=_Retrieval(r),
        prompts=_Prompts(),
        llm=_LLM("ignored"),
        citations=_Citations([]),
        run_logger=_RunLogger(run_repo),
        conversations_repo=convo,
        messages_repo=msgs,
    )

    res = await svc.answer(ctx=tenant_ctx, req=ChatRequest(message="hola", mode=ChatMode.strict))

    assert res.confidence == ConfidenceLevel.low
    assert res.citations == []
    assert "enough evidence" in res.answer
    assert len(convo.created) == 1
    assert msgs.messages[0][1] == "user"
    assert msgs.messages[1][1] == "assistant"


@pytest.mark.asyncio
async def test_chat_service_strict_requires_valid_citations(tenant_ctx):
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id=uuid4(),
        title="T",
        content="x",
        score=0.9,
        metadata={},
    )
    r = RetrievalResult(chunks=[chunk], evidence_strength=0.9, debug={})

    svc = ChatService(
        retrieval=_Retrieval(r),
        prompts=_Prompts(),
        llm=_LLM("Respuesta sin citas"),
        citations=_Citations([], strict_ok=False),
        run_logger=_RunLogger(_RunRepo()),
        conversations_repo=_ConversationsRepo(),
        messages_repo=_MessagesRepo(),
    )

    res = await svc.answer(ctx=tenant_ctx, req=ChatRequest(message="q", mode=ChatMode.strict))
    assert res.citations == []
    assert res.confidence == ConfidenceLevel.low
    assert "enough evidence" in res.answer


@pytest.mark.asyncio
async def test_chat_service_happy_path_returns_llm_answer(tenant_ctx):
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id=uuid4(),
        title="T",
        content="x",
        score=0.95,
        metadata={},
    )
    r = RetrievalResult(chunks=[chunk], evidence_strength=0.95, debug={})

    svc = ChatService(
        retrieval=_Retrieval(r),
        prompts=_Prompts(),
        llm=_LLM("Ok [C1]"),
        citations=_Citations(
            [
                Citation(
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    chunk_id=chunk.chunk_id,
                    snippet="x",
                    section=None,
                    page=None,
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
            ],
            strict_ok=True,
        ),
        run_logger=_RunLogger(_RunRepo()),
        conversations_repo=_ConversationsRepo(),
        messages_repo=_MessagesRepo(),
    )

    res = await svc.answer(ctx=tenant_ctx, req=ChatRequest(message="q", mode=ChatMode.normal))
    assert res.answer == "Ok [C1]"
    assert res.citations != []
    assert res.retrieval_debug is not None


@pytest.mark.asyncio
async def test_chat_service_propagates_llm_timeout_and_does_not_persist_run(tenant_ctx):
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id=uuid4(),
        title="T",
        content="x",
        score=0.95,
        metadata={},
    )
    r = RetrievalResult(chunks=[chunk], evidence_strength=0.95, debug={})

    convo = _ConversationsRepo()
    msgs = _MessagesRepo()
    run_repo = _RunRepo()

    svc = ChatService(
        retrieval=_Retrieval(r),
        prompts=_Prompts(),
        llm=_LLMTimeout(),
        citations=_Citations([], strict_ok=True),
        run_logger=_RunLogger(run_repo),
        conversations_repo=convo,
        messages_repo=msgs,
    )

    with pytest.raises(TimeoutError):
        await svc.answer(ctx=tenant_ctx, req=ChatRequest(message="q", mode=ChatMode.normal))

    # Se crea conversación y se inserta el mensaje del usuario, pero no el del asistente.
    assert len(convo.created) == 1
    assert len(msgs.messages) == 1
    assert msgs.messages[0][1] == "user"

    # No se persiste run porque la generación falló.
    assert run_repo.inserted == []
