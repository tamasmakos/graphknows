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

# Bump when a prompt change should invalidate cached verdicts. Bumped for the
# two-stage judge: the headline score is now the STAGE-2 verdict (not raw
# stage-1 SemanticF1), so old single-stage cache entries must not be reused.
# v3: date tolerance removed from the judge (strict same-day matching). The
# version string keys the judge cache — bumping it is what invalidates verdicts
# cached under the lenient prompt; without the bump they replay silently.
_PROMPT_VERSION = "locomo-judge-two-stage-v3-strict-dates"


def render_facts(facts: list[str] | None) -> str:
    """Render the D8 fact-sheet as a prompt block, or "" when there are none.

    Empty facts yield "" so the formatted prompt is byte-identical to the
    pre-D8 baseline (the ``{facts}`` slot sits immediately before "Memories"),
    keeping the GRAPHKNOWS_FACT_CONTEXT=off arm a true baseline.
    """
    if not facts:
        return ""
    lines = "\n".join(f"- {f}" for f in facts)
    return f"Known facts (structured, extracted from the conversations):\n{lines}\n\n"


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
        self,
        case: EvalCase,
        unit: IngestUnit,
        passages: list[RetrievedPassage],
        facts: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
            reference_date=unit.reference_date or "2023",
            facts=render_facts(facts),
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
        # NOTE: the headline `score` is now the STAGE-2 verdict from the two-stage
        # judge, not the raw stage-1 SemanticF1. This intentionally redefines the
        # LoCoMo score (baseline reset expected). Stage-1 `semantic_f1` is still
        # recorded alongside it so the two can be compared per case.
        gold = _judge_gold(case)
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
        return {
            # Official paper-comparable token-F1 (category-aware, stemmed).
            "locomo_f1": locomo_f1(case.gold, answer, case.category),
            "token_f1": token_f1(case.gold, answer),
        }
