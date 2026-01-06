from dataclasses import dataclass
from uuid import UUID, uuid4

from packages.shared.schemas.chat import ChatMode, ChatRequest, ChatResponse
from packages.shared.schemas.common import ConfidenceLevel, TenantContext

from .citation_service import CitationService
from .llm_client import LLMClient
from .prompt_service import PromptService
from .retrieval_service import RetrievalService
from .run_logger import RunLogInput, RunLogger


@dataclass(slots=True, frozen=True)
class ChatPolicy:
    min_evidence_strength_normal: float = 0.25
    min_evidence_strength_strict: float = 0.35


class ChatService:
    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        prompts: PromptService,
        llm: LLMClient,
        citations: CitationService,
        run_logger: RunLogger,
        conversations_repo,
        messages_repo,
        policy: ChatPolicy = ChatPolicy(),
        default_model: str = "gpt-4.1-mini",
    ) -> None:
        self._retrieval = retrieval
        self._prompts = prompts
        self._llm = llm
        self._citations = citations
        self._run_logger = run_logger
        self._conversations_repo = conversations_repo
        self._messages_repo = messages_repo
        self._policy = policy
        self._default_model = default_model

    async def answer(self, *, ctx: TenantContext, req: ChatRequest) -> ChatResponse:
        conversation_id = req.conversation_id or uuid4()
        if req.conversation_id is None:
            await self._conversations_repo.create(conversation_id, ctx)

        await self._messages_repo.insert(conversation_id, role="user", content=req.message)

        r = await self._retrieval.retrieve(
            ctx=ctx,
            question=req.message,
            filters=req.filters,
            top_k=req.top_k,
            use_rerank=req.use_rerank,
        )

        threshold = (
            self._policy.min_evidence_strength_strict
            if req.mode == ChatMode.strict
            else self._policy.min_evidence_strength_normal
        )

        if r.evidence_strength < threshold or not r.chunks:
            answer = (
                "I don’t have enough evidence in the provided documents to answer that.\n"
                "Try adding more documentation or narrowing the question."
                if req.mode == ChatMode.strict
                else
                "I don’t have enough information in the available documents to answer confidently.\n"
                "Can you clarify which policy/process or department this refers to?"
            )
            await self._messages_repo.insert(conversation_id, role="assistant", content=answer)

            run_id = await self._run_logger.persist(
                RunLogInput(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    conversation_id=conversation_id,
                    question=req.message,
                    answer=answer,
                    model="(no-llm)",
                    usage=None,
                    retrieved_doc_ids=[str(c.doc_id) for c in r.chunks],
                    retrieval_debug=r.debug,
                )
            )

            return ChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                citations=[],
                confidence=ConfidenceLevel.low,
                follow_ups=[
                    "Which document or policy should I use as a reference?",
                    "Which department/process is this about?",
                    "Can you share an example or a keyword from the document?",
                ],
                run_id=run_id,
                usage=None,
                retrieval_debug=r.debug,
            )

        system = self._prompts.system_prompt(req.mode)
        context = self._prompts.build_context(r.chunks)
        user = req.message

        model = self._default_model
        llm_res = await self._llm.generate(system=system, user=user, context=context, model=model)

        citations = self._citations.build_citations(r.chunks, llm_res.text)

        if req.mode == ChatMode.strict and not self._citations.validate_strict(llm_res.text, citations):
            answer = (
                "I don’t have enough evidence in the provided documents to answer that.\n"
                "Please provide more documentation or rephrase the question."
            )
            await self._messages_repo.insert(conversation_id, role="assistant", content=answer)

            run_id = await self._run_logger.persist(
                RunLogInput(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    conversation_id=conversation_id,
                    question=req.message,
                    answer=answer,
                    model=model,
                    usage=llm_res.usage,
                    retrieved_doc_ids=[str(c.doc_id) for c in r.chunks],
                    retrieval_debug=r.debug,
                )
            )

            return ChatResponse(
                conversation_id=conversation_id,
                answer=answer,
                citations=[],
                confidence=ConfidenceLevel.low,
                follow_ups=[
                    "Can you point me to the exact doc/section?",
                    "Do you have a PDF or link to ingest?",
                ],
                run_id=run_id,
                usage=llm_res.usage,
                retrieval_debug=r.debug,
            )

        await self._messages_repo.insert(conversation_id, role="assistant", content=llm_res.text)

        if r.evidence_strength >= threshold + 0.15:
            conf = ConfidenceLevel.high
        elif r.evidence_strength >= threshold + 0.05:
            conf = ConfidenceLevel.medium
        else:
            conf = ConfidenceLevel.low

        run_id = await self._run_logger.persist(
            RunLogInput(
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                conversation_id=conversation_id,
                question=req.message,
                answer=llm_res.text,
                model=model,
                usage=llm_res.usage,
                retrieved_doc_ids=[str(c.doc_id) for c in r.chunks],
                retrieval_debug=r.debug,
            )
        )

        return ChatResponse(
            conversation_id=conversation_id,
            answer=llm_res.text,
            citations=citations,
            confidence=conf,
            follow_ups=[
                "Do you want a step-by-step version?",
                "Should I summarise the relevant section and link sources?",
                "Do you want the policy differences by region/team?",
            ],
            run_id=run_id,
            usage=llm_res.usage,
            retrieval_debug=r.debug,
        )
