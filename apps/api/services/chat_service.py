from dataclasses import dataclass
from uuid import uuid4

from packages.shared.schemas.chat import ChatMode, ChatRequest, ChatResponse, RefusalReason
from packages.shared.schemas.common import ConfidenceLevel, TenantContext

from .citation_service import CitationService
from .llm_client import LLMClient
from .observability import get_meter, get_tracer
from .prompt_service import PromptService
from .retrieval_service import RetrievalService
from .run_logger import RunLogInput, RunLogger


@dataclass(slots=True, frozen=True)
class ChatPolicy:
    min_evidence_strength_normal: float = 0.25
    min_evidence_strength_strict: float = 0.35

    # Rechazo canónico (sin filtrar info sensible)
    strict_refusal_text: str = (
        "I don’t have enough evidence in the provided documents to answer that."
    )
    strict_refusal_reason: RefusalReason = RefusalReason.insufficient_grounding_or_citations


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

        # Observabilidad
        self._tracer = get_tracer("apps.api.chat")
        self._meter = get_meter("apps.api.chat")

        self._chat_requests = self._meter.create_counter(
            "chat_requests_total",
            description="Total de requests de chat",
        )
        self._chat_latency_ms = self._meter.create_histogram(
            "chat_latency_ms",
            unit="ms",
            description="Latencia total de answer()",
        )
        self._strict_refusals = self._meter.create_counter(
            "chat_strict_refusals_total",
            description="Rechazos en modo strict",
        )
        self._llm_tokens = self._meter.create_counter(
            "llm_tokens_total",
            description="Tokens totales por modelo/tenant",
        )
        self._llm_cost_usd = self._meter.create_counter(
            "llm_cost_usd_total",
            unit="USD",
            description="Coste estimado acumulado por modelo/tenant",
        )

    async def answer(self, *, ctx: TenantContext, req: ChatRequest) -> ChatResponse:
        import time

        t0 = time.perf_counter()
        conversation_id = req.conversation_id or uuid4()

        attrs = {
            "tenant_id": str(ctx.tenant_id),
            "user_id": str(ctx.user_id),
            "chat.mode": str(req.mode),
            "chat.top_k": int(req.top_k),
            "chat.use_rerank": bool(req.use_rerank),
            "conversation_id": str(conversation_id),
        }
        self._chat_requests.add(1, attributes=attrs)

        with self._tracer.start_as_current_span("chat.answer", attributes=attrs) as span:
            if req.conversation_id is None:
                with self._tracer.start_as_current_span("chat.conversation.create"):
                    await self._conversations_repo.create(conversation_id, ctx)

            with self._tracer.start_as_current_span("chat.message.insert", attributes={"role": "user"}):
                await self._messages_repo.insert(conversation_id, role="user", content=req.message)

            with self._tracer.start_as_current_span("retrieval.retrieve") as rspan:
                r = await self._retrieval.retrieve(
                    ctx=ctx,
                    question=req.message,
                    filters=req.filters,
                    top_k=req.top_k,
                    use_rerank=req.use_rerank,
                )
                rspan.set_attribute("retrieval.selected_n", int(len(r.chunks)))
                rspan.set_attribute("retrieval.evidence_strength", float(r.evidence_strength))
                rspan.set_attribute("retrieval.doc_ids", [str(c.doc_id) for c in r.chunks])
                rspan.set_attribute("retrieval.chunk_ids", [str(c.chunk_id) for c in r.chunks])

            threshold = (
                self._policy.min_evidence_strength_strict
                if req.mode == ChatMode.strict
                else self._policy.min_evidence_strength_normal
            )
            span.set_attribute("policy.threshold", float(threshold))

            if r.evidence_strength < threshold or not r.chunks:
                answer = (
                    "I don’t have enough evidence in the provided documents to answer that.\n"
                    "Try adding more documentation or narrowing the question."
                    if req.mode == ChatMode.strict
                    else "I don’t have enough information in the available documents to answer confidently.\n"
                    "Can you clarify which policy/process or department this refers to?"
                )

                with self._tracer.start_as_current_span("chat.message.insert", attributes={"role": "assistant"}):
                    await self._messages_repo.insert(conversation_id, role="assistant", content=answer)

                refusal_reason = (
                    self._policy.strict_refusal_reason if req.mode == ChatMode.strict else None
                )

                if req.mode == ChatMode.strict:
                    self._strict_refusals.add(1, attributes={**attrs, "reason": refusal_reason.value})
                    span.set_attribute("chat.refused", True)
                    span.set_attribute("chat.refusal_reason", refusal_reason.value)

                debug = (
                    {**(r.debug or {}), "strict_refusal_reason": refusal_reason.value}
                    if refusal_reason is not None
                    else (r.debug or {})
                )
                debug = dict(debug or {})
                debug["used_chunk_ids"] = []
                debug["used_doc_ids"] = []

                with self._tracer.start_as_current_span("runs.persist"):
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
                            retrieval_debug=debug,
                        )
                    )
                span.set_attribute("run_id", str(run_id))

                self._chat_latency_ms.record(int((time.perf_counter() - t0) * 1000), attributes=attrs)

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
                    refused=req.mode == ChatMode.strict,
                    refusal_reason=refusal_reason,
                    run_id=run_id,
                    usage=None,
                    retrieval_debug=debug,
                )

            system = self._prompts.system_prompt(req.mode)
            context = self._prompts.build_context(r.chunks)
            user = req.message

            model = self._default_model
            with self._tracer.start_as_current_span(
                "llm.generate",
                attributes={"tenant_id": str(ctx.tenant_id), "llm.model": model, "chat.mode": str(req.mode)},
            ):
                llm_res = await self._llm.generate(system=system, user=user, context=context, model=model)

            if llm_res.usage is not None:
                try:
                    self._llm_tokens.add(
                        int(getattr(llm_res.usage, "total_tokens", 0) or 0),
                        attributes={"tenant_id": str(ctx.tenant_id), "model": model},
                    )
                    self._llm_cost_usd.add(
                        float(getattr(llm_res.usage, "cost_estimate_usd", 0.0) or 0.0),
                        attributes={"tenant_id": str(ctx.tenant_id), "model": model},
                    )
                except Exception:
                    pass

            with self._tracer.start_as_current_span("citations.build"):
                citations = self._citations.build_citations(r.chunks, llm_res.text)

            used_chunk_ids = [c.chunk_id for c in citations]
            used_doc_ids = list(dict.fromkeys(str(c.doc_id) for c in citations))
            span.set_attribute("citations.count", int(len(citations)))
            span.set_attribute("citations.used_chunk_ids", [str(x) for x in used_chunk_ids])
            span.set_attribute("citations.used_doc_ids", used_doc_ids)

            if req.mode == ChatMode.strict:
                with self._tracer.start_as_current_span("citations.validate_strict"):
                    strict_ok = self._citations.validate_strict(
                        llm_res.text,
                        citations,
                        retrieved_chunk_count=len(r.chunks),
                    )

                if not strict_ok:
                    answer = (
                        f"{self._policy.strict_refusal_text}\n"
                        "Please provide more documentation or rephrase the question."
                    )
                    await self._messages_repo.insert(conversation_id, role="assistant", content=answer)

                    self._strict_refusals.add(
                        1,
                        attributes={**attrs, "reason": self._policy.strict_refusal_reason.value},
                    )
                    span.set_attribute("chat.refused", True)
                    span.set_attribute("chat.refusal_reason", self._policy.strict_refusal_reason.value)

                    debug = {
                        **(r.debug or {}),
                        "strict_refusal_reason": self._policy.strict_refusal_reason.value,
                        "used_chunk_ids": used_chunk_ids,
                        "used_doc_ids": used_doc_ids,
                    }

                    with self._tracer.start_as_current_span("runs.persist"):
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
                                retrieval_debug=debug,
                            )
                        )

                    span.set_attribute("run_id", str(run_id))
                    self._chat_latency_ms.record(int((time.perf_counter() - t0) * 1000), attributes=attrs)

                    return ChatResponse(
                        conversation_id=conversation_id,
                        answer=answer,
                        citations=[],
                        confidence=ConfidenceLevel.low,
                        follow_ups=[
                            "Can you point me to the exact doc/section?",
                            "Do you have a PDF or link to ingest?",
                        ],
                        refused=True,
                        refusal_reason=self._policy.strict_refusal_reason,
                        run_id=run_id,
                        usage=llm_res.usage,
                        retrieval_debug=debug,
                    )

            await self._messages_repo.insert(conversation_id, role="assistant", content=llm_res.text)

            if r.evidence_strength >= threshold + 0.15:
                conf = ConfidenceLevel.high
            elif r.evidence_strength >= threshold + 0.05:
                conf = ConfidenceLevel.medium
            else:
                conf = ConfidenceLevel.low

            debug = dict(r.debug or {})
            debug["used_chunk_ids"] = used_chunk_ids
            debug["used_doc_ids"] = used_doc_ids

            with self._tracer.start_as_current_span("runs.persist"):
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
                        retrieval_debug=debug,
                    )
                )

            span.set_attribute("run_id", str(run_id))
            span.set_attribute("chat.refused", False)

            self._chat_latency_ms.record(int((time.perf_counter() - t0) * 1000), attributes=attrs)

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
                retrieval_debug=debug,
            )
