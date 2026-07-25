"""LongMemEval adapter: staged-reasoning answerer + shared two-stage DSPy judge."""

from __future__ import annotations

import logging
from typing import Any

from evaluation.common.cache import JudgeCache
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.lexical import token_f1
from evaluation.common.llm import extract_answer
from evaluation.common.memories import format_memories
from evaluation.common.pipeline import BenchmarkAdapter
from evaluation.longmem import prompts

log = logging.getLogger(__name__)

# Bump when a prompt change should invalidate cached verdicts. v2: switched to
# the two-stage judge (headline score is the STAGE-2 verdict, not raw stage-1
# SemanticF1), matching LoCoMo — old single-stage cache entries must not replay.
_PROMPT_VERSION = "longmem-judge-two-stage-v2"


class LongMemAdapter(BenchmarkAdapter):
    name = "longmem"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache = JudgeCache("longmem")

    async def answer(
        self,
        case: EvalCase,
        unit: IngestUnit,
        passages: list[RetrievedPassage],
        facts: list[str] | None = None,  # D8 fact-sheet — unused by longmem
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
            reference_date=unit.reference_date or case.question_date or "2023",
            memories=format_memories(passages),
            question=case.question,
        )
        text, usage = await self.answerer.generate(prompts.ANSWER_SYSTEM, prompt)
        answer = extract_answer(text) or "Unknown"
        return answer, {
            "gen_prompt_tokens": usage["prompt_tokens"],
            "gen_completion_tokens": usage["completion_tokens"],
            "gen_total_tokens": usage["total_tokens"],
            "gen_cost_usd": usage["cost_usd"],
        }

    async def judge_case(self, case: EvalCase, answer: str) -> tuple[float, dict[str, Any]]:
        # Headline `score` is the STAGE-2 verdict from the two-stage judge (same
        # as LoCoMo), so LongMemEval accuracy is measured identically. Stage-1
        # `semantic_f1` and lexical `token_f1` are recorded alongside it.
        gold = case.gold
        cached = self._cache.get(_PROMPT_VERSION, case.question, gold, answer)
        if cached is not None:
            return float(cached["score"]), {
                "judge_label": cached.get("label", ""),
                "judge_reasoning": cached.get("reasoning", ""),
                "semantic_f1": cached.get("semantic_f1"),
                "token_f1": cached.get("token_f1"),
                "judge_cached": True,
            }

        try:
            score, jm = await self.judge.judge_two_stage(case.question, gold, answer)
        except Exception as exc:
            log.exception("Judge failed for %s (scored 0.0)", case.case_id)
            return 0.0, {"judge_label": "ERROR", "judge_reasoning": str(exc)[:200]}

        label = "CORRECT" if score >= 0.5 else "WRONG"
        reasoning = jm.get("judge_reasoning", "")
        self._cache.put(
            {
                "score": score,
                "label": label,
                "reasoning": reasoning,
                "semantic_f1": jm.get("semantic_f1"),
                "token_f1": jm.get("token_f1"),
            },
            _PROMPT_VERSION,
            case.question,
            gold,
            answer,
        )
        return score, {
            "judge_label": label,
            "judge_reasoning": reasoning,
            "semantic_f1": jm.get("semantic_f1"),
            "token_f1": jm.get("token_f1"),
        }

    def lexical_metrics(self, case: EvalCase, answer: str) -> dict[str, Any]:
        return {"token_f1": token_f1(case.gold, answer)}
