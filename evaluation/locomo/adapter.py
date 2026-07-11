"""LoCoMo benchmark adapter: staged-reasoning answerer + binary CORRECT/WRONG judge."""

from __future__ import annotations

import logging
from typing import Any

from evaluation.common.cache import JudgeCache
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.lexical import locomo_f1, token_f1
from evaluation.common.llm import extract_answer
from evaluation.common.memories import format_memories
from evaluation.common.pipeline import BenchmarkAdapter
from evaluation.locomo import prompts

log = logging.getLogger(__name__)

# Bump when a prompt change should invalidate cached verdicts.
_PROMPT_VERSION = "locomo-judge-v1"


def _judge_gold(case: EvalCase) -> str:
    """Gold preprocessing for the judge: open-ended golds keep only the text
    before the first ';' (the remainder is commentary, not the answer)."""
    if case.category == "open_ended":
        return case.gold.split(";")[0].strip()
    return case.gold


class LocomoAdapter(BenchmarkAdapter):
    name = "locomo"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache = JudgeCache("locomo")

    async def answer(
        self, case: EvalCase, unit: IngestUnit, passages: list[RetrievedPassage]
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
            reference_date=unit.reference_date or "2023",
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
        gold = _judge_gold(case)
        cached = self._cache.get(_PROMPT_VERSION, case.question, gold, answer)
        if cached is not None:
            return float(cached["score"]), {
                "judge_label": cached.get("label", ""),
                "judge_reasoning": cached.get("reasoning", ""),
                "judge_cached": True,
            }

        prompt = prompts.JUDGE_PROMPT.format(question=case.question, gold=gold, answer=answer)
        try:
            verdict, usage = await self.judge.generate_json(prompts.JUDGE_SYSTEM, prompt)
        except Exception as exc:
            log.exception("Judge failed for %s (scored 0.0)", case.case_id)
            return 0.0, {"judge_label": "ERROR", "judge_reasoning": str(exc)[:200]}

        label = str(verdict.get("label", "")).strip().upper()
        score = 1.0 if label == "CORRECT" else 0.0
        reasoning = str(verdict.get("reasoning", ""))
        self._cache.put(
            {"score": score, "label": label, "reasoning": reasoning},
            _PROMPT_VERSION, case.question, gold, answer,
        )
        return score, {
            "judge_label": label or "UNPARSEABLE",
            "judge_reasoning": reasoning,
            "judge_total_tokens": usage["total_tokens"],
            "judge_cost_usd": usage["cost_usd"],
        }

    def lexical_metrics(self, case: EvalCase, answer: str) -> dict[str, Any]:
        return {
            # Official paper-comparable token-F1 (category-aware, stemmed).
            "locomo_f1": locomo_f1(case.gold, answer, case.category),
            "token_f1": token_f1(case.gold, answer),
        }
