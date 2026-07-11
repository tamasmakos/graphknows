"""LongMemEval benchmark adapter: dated-memory answerer + yes/no judge."""

from __future__ import annotations

import logging
import re
from typing import Any

from evaluation.common.cache import JudgeCache
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.llm import extract_answer
from evaluation.common.memories import format_memories
from evaluation.common.pipeline import BenchmarkAdapter
from evaluation.longmemeval import prompts

log = logging.getLogger(__name__)

_PROMPT_VERSION = "longmemeval-judge-v2"
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    """Parse the judge's final yes/no verdict.

    Takes the region after the last closing think tag, scans lines bottom-up
    for a bare yes/no, then falls back to the last yes/no token anywhere.
    Returns None when no verdict is found.
    """
    region = text or ""
    for tag in ("</judge_thinking>", "</thinking>"):
        idx = region.rfind(tag)
        if idx >= 0:
            region = region[idx + len(tag) :]
            break
    for line in reversed(region.strip().splitlines()):
        token = line.strip().strip(".!\"'*").lower()
        if token in ("yes", "no"):
            return token == "yes"
    matches = _YES_NO_RE.findall(region)
    if matches:
        return matches[-1].lower() == "yes"
    return None


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache = JudgeCache("longmemeval")

    async def answer(
        self, case: EvalCase, unit: IngestUnit, passages: list[RetrievedPassage]
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
            question_date=case.question_date or "unknown",
            memories=format_memories(passages),
            question=case.question,
        )
        # Reasoning happens in-band (<mem_thinking>), so give the completion
        # extra headroom beyond the answerer default.
        text, usage = await self.answerer.generate(
            prompts.ANSWER_SYSTEM, prompt, max_tokens=8192
        )
        answer = extract_answer(text) or "The information provided is not enough"
        return answer, {
            "gen_prompt_tokens": usage["prompt_tokens"],
            "gen_completion_tokens": usage["completion_tokens"],
            "gen_total_tokens": usage["total_tokens"],
            "gen_cost_usd": usage["cost_usd"],
        }

    async def judge_case(self, case: EvalCase, answer: str) -> tuple[float, dict[str, Any]]:
        cached = self._cache.get(_PROMPT_VERSION, case.question, case.gold, answer)
        if cached is not None:
            return float(cached["score"]), {
                "judge_label": cached.get("label", ""),
                "judge_cached": True,
            }

        prompt = prompts.JUDGE_PROMPT.format(
            question=case.question, gold=case.gold, answer=answer
        )
        try:
            text, usage = await self.judge.generate("", prompt, max_tokens=4096)
        except Exception as exc:
            log.exception("Judge failed for %s (scored 0.0)", case.case_id)
            return 0.0, {"judge_label": "ERROR", "judge_reasoning": str(exc)[:200]}

        verdict = parse_yes_no(text)
        score = 1.0 if verdict else 0.0
        label = "yes" if verdict else ("no" if verdict is not None else "UNPARSEABLE")
        self._cache.put(
            {"score": score, "label": label},
            _PROMPT_VERSION, case.question, case.gold, answer,
        )
        return score, {
            "judge_label": label,
            "judge_total_tokens": usage["total_tokens"],
            "judge_cost_usd": usage["cost_usd"],
        }
