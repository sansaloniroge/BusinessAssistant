from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from packages.shared.schemas.eval import JudgeInput, JudgeOutput

from apps.api.services.llm_client import LLMClient


@dataclass(slots=True, frozen=True)
class JudgePolicy:
    # Modelo por defecto del judge (puede ser distinto al del chat)
    default_judge_model: str = "gpt-4.1-mini"


class EvalJudgeService:
    """LLM-as-judge.

    Diseñado para evaluar runs reales: recibe un JudgeInput que proviene de un run persistido
    (question/answer/retrieval_debug/citations) y devuelve un JudgeOutput con scores 0..5.

    Importante: aquí NO se vuelve a ejecutar retrieval; se juzga lo que pasó.
    """

    def __init__(self, *, llm: LLMClient, policy: JudgePolicy = JudgePolicy()) -> None:
        self._llm = llm
        self._policy = policy

    def _system_prompt(self) -> str:
        return (
            "You are an impartial evaluator for an enterprise RAG assistant.\n"
            "You will receive JSON with: question, answer, citations, retrieved_doc_ids, retrieval_debug, mode.\n"
            "Score each dimension from 0 to 5 (integers):\n"
            "- faithfulness: answer supported by provided citations/context; penalize hallucinations.\n"
            "- relevance: answers the user question.\n"
            "- citation_quality: citations are specific, correctly referenced, and sufficient for key claims.\n"
            "- refusal_correctness: if the assistant refused, was refusal appropriate given evidence?\n"
            "- overall: overall quality (not a simple average; must be consistent with above).\n"
            "Return STRICT JSON only with keys: overall, faithfulness, relevance, citation_quality, refusal_correctness, rationale.\n"
            "Do not include extra keys or markdown.\n"
        )

    def _user_prompt(self, inp: JudgeInput) -> str:
        # mode='json' convierte UUID/datetime/etc a tipos JSON-serializables.
        payload = inp.model_dump(mode="json")
        return json.dumps(payload, ensure_ascii=False)

    async def judge(
        self,
        *,
        inp: JudgeInput,
        model: str | None = None,
    ) -> tuple[JudgeOutput, Any, str]:
        """Evalúa un run y devuelve (output, usage, judge_model_efectivo)."""

        judge_model = model or self._policy.default_judge_model
        res = await self._llm.generate(
            system=self._system_prompt(),
            user=self._user_prompt(inp),
            context="",
            model=judge_model,
        )

        raw = res.text.strip()
        try:
            data = json.loads(raw)
        except Exception:
            # Fallback ultra-defensivo: si el LLM no devuelve JSON, degradar a 0 con rationale crudo.
            out = JudgeOutput(
                overall=0,
                faithfulness=0,
                relevance=0,
                citation_quality=0,
                refusal_correctness=0,
                rationale=f"Invalid judge output (non-JSON): {raw[:1000]}",
            )
            return out, res.usage, judge_model

        def clamp_int(v: Any) -> int:
            try:
                i = int(v)
            except Exception:
                i = 0
            return max(0, min(5, i))

        out = JudgeOutput(
            overall=clamp_int(data.get("overall")),
            faithfulness=clamp_int(data.get("faithfulness")),
            relevance=clamp_int(data.get("relevance")),
            citation_quality=clamp_int(data.get("citation_quality")),
            refusal_correctness=clamp_int(data.get("refusal_correctness")),
            rationale=str(data.get("rationale") or ""),
        )
        return out, res.usage, judge_model
