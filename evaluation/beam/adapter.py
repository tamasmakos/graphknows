"""BEAM adapter: staged-reasoning answerer + category-routed DSPy judging.

Eight categories have a gold answer → the shared SemanticF1 judge. The two
compliance categories (instruction/preference following) have no gold string →
the shared rubric judge, grading against a composed compliance rubric.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from evaluation.beam import prompts
from evaluation.common.cache import JudgeCache
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.llm import extract_answer
from evaluation.common.memories import format_memories
from evaluation.common.pipeline import BenchmarkAdapter

log = logging.getLogger(__name__)

# Bump when a prompt change should invalidate cached verdicts.
_PROMPT_VERSION = "beam-judge-dspy-v1"

# Categories graded against a compliance rubric rather than a gold string.
_RUBRIC_CATEGORIES = frozenset({"instruction_following", "preference_following"})


def _stringify(value: Any) -> str:
    """Flatten a rubric/indicator value (str | list | dict) into readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(_stringify(v) for v in value if v)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_stringify(v)}" for k, v in value.items() if v)
    return str(value)


def _compose_rubric(case: EvalCase) -> str:
    """Build the grading rubric text for a compliance question from its extras."""
    e = case.extras
    parts: list[str] = []
    if e.get("being_tested"):
        parts.append(f"Instruction/preference under test: {_stringify(e['being_tested'])}")
    if e.get("expected_compliance"):
        parts.append(f"Expected compliance: {_stringify(e['expected_compliance'])}")
    if e.get("rubric"):
        parts.append(f"Rubric: {_stringify(e['rubric'])}")
    if e.get("compliance_indicators"):
        parts.append(f"Signs of compliance: {_stringify(e['compliance_indicators'])}")
    if e.get("non_compliance_signs"):
        parts.append(f"Signs of non-compliance: {_stringify(e['non_compliance_signs'])}")
    return "\n".join(parts) or "Answer honors the tested instruction/preference."


class BeamAdapter(BenchmarkAdapter):
    name = "beam"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache = JudgeCache("beam")

    async def answer(
        self,
        case: EvalCase,
        unit: IngestUnit,
        passages: list[RetrievedPassage],
        facts: list[str] | None = None,  # D8 fact-sheet — unused by beam
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
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
        is_rubric = case.category in _RUBRIC_CATEGORIES
        reference = _compose_rubric(case) if is_rubric else case.gold

        cached = self._cache.get(_PROMPT_VERSION, case.question, reference, answer)
        if cached is not None:
            return float(cached["score"]), {
                "judge_label": cached.get("label", ""),
                "judge_reasoning": cached.get("reasoning", ""),
                "judge_mode": "rubric" if is_rubric else "gold",
                "judge_cached": True,
            }

        try:
            if is_rubric:
                score, reasoning = await self.judge.judge_rubric(case.question, reference, answer)
            else:
                score, reasoning = await self.judge.judge(case.question, reference, answer)
        except Exception as exc:
            log.exception("Judge failed for %s (scored 0.0)", case.case_id)
            return 0.0, {"judge_label": "ERROR", "judge_reasoning": str(exc)[:200]}

        label = "CORRECT" if score >= 0.5 else "WRONG"
        self._cache.put(
            {"score": score, "label": label, "reasoning": reasoning},
            _PROMPT_VERSION,
            case.question,
            reference,
            answer,
        )
        return score, {
            "judge_label": label,
            "judge_reasoning": reasoning,
            "judge_mode": "rubric" if is_rubric else "gold",
        }

    def lexical_metrics(self, case: EvalCase, answer: str) -> dict[str, Any]:
        # source-chat provenance is handy in the per-case CSV for error analysis
        src = case.extras.get("source_chat_ids")
        return {"source_chat_ids": json.dumps(src)} if src else {}
