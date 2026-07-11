"""BEAM benchmark adapter: rule-based answerer + per-nugget rubric judge."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from evaluation.beam import prompts
from evaluation.common.cache import JudgeCache
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.lexical import kendall_tau_b
from evaluation.common.llm import extract_answer
from evaluation.common.memories import format_memories
from evaluation.common.pipeline import BenchmarkAdapter

log = logging.getLogger(__name__)

_PROMPT_VERSION = "beam-judge-v1"
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def clamp_nugget_score(raw: Any) -> float:
    """Snap a judge score onto the {0.0, 0.5, 1.0} rubric scale."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value >= 0.75:
        return 1.0
    if value >= 0.25:
        return 0.5
    return 0.0


def _parse_json_array(text: str) -> list[str]:
    for candidate in (text, *( [m.group(0)] if (m := _ARRAY_RE.search(text or "")) else [] )):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


class BeamAdapter(BenchmarkAdapter):
    name = "beam"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache = JudgeCache("beam")

    async def answer(
        self, case: EvalCase, unit: IngestUnit, passages: list[RetrievedPassage]
    ) -> tuple[str, dict[str, Any]]:
        prompt = prompts.ANSWER_PROMPT.format(
            question=case.question,
            memories=format_memories(passages),
        )
        text, usage = await self.answerer.generate(prompts.ANSWER_SYSTEM, prompt)
        answer = extract_answer(text) or "I don't have enough information to answer this question."
        return answer, {
            "gen_prompt_tokens": usage["prompt_tokens"],
            "gen_completion_tokens": usage["completion_tokens"],
            "gen_total_tokens": usage["total_tokens"],
            "gen_cost_usd": usage["cost_usd"],
        }

    async def _judge_nugget(self, case: EvalCase, answer: str, nugget: str) -> float:
        cached = self._cache.get(_PROMPT_VERSION, case.question, nugget, answer)
        if cached is not None:
            return float(cached["score"])
        prompt = prompts.NUGGET_JUDGE_PROMPT.format(
            question=case.question, answer=answer, nugget=nugget
        )
        try:
            verdict, _ = await self.judge.generate_json(prompts.NUGGET_JUDGE_SYSTEM, prompt)
        except Exception:
            log.exception("Nugget judge failed for %s (scored 0.0)", case.case_id)
            return 0.0
        score = clamp_nugget_score(verdict.get("score"))
        self._cache.put({"score": score}, _PROMPT_VERSION, case.question, nugget, answer)
        return score

    async def _event_ordering_tau(self, case: EvalCase, answer: str) -> float:
        """Kendall tau-b between the answer's event order and the rubric order."""
        nuggets = case.extras.get("nuggets", [])
        if len(nuggets) < 2:
            return 0.0
        extraction = prompts.EVENT_EXTRACTION_PROMPT.format(answer=answer)
        text, _ = await self.judge.generate(prompts.EVENT_EXTRACTION_SYSTEM, extraction)
        events = _parse_json_array(text)
        events_list = "\n".join(f"{i}: {n}" for i, n in enumerate(nuggets))
        aligned: list[int] = []
        for event in events:
            alignment = prompts.EVENT_ALIGNMENT_PROMPT.format(
                event=event, events_list=events_list
            )
            try:
                verdict, _ = await self.judge.generate_json(
                    prompts.EVENT_ALIGNMENT_SYSTEM, alignment
                )
                index = int(verdict.get("index", -1))
            except Exception:
                index = -1
            if 0 <= index < len(nuggets) and index not in aligned:
                aligned.append(index)
        if len(aligned) < 2:
            return -1.0 if events else 0.0
        return kendall_tau_b(aligned, sorted(aligned))

    async def judge_case(self, case: EvalCase, answer: str) -> tuple[float, dict[str, Any]]:
        nuggets: list[str] = case.extras.get("nuggets") or ([case.gold] if case.gold else [])
        if not nuggets:
            return 0.0, {"judge_label": "NO_RUBRIC"}

        nugget_scores = [await self._judge_nugget(case, answer, n) for n in nuggets]
        nugget_mean = sum(nugget_scores) / len(nugget_scores)

        metrics: dict[str, Any] = {
            "nugget_mean": round(nugget_mean, 4),
            "nugget_count": len(nuggets),
            "nugget_scores": nugget_scores,
        }
        score = nugget_mean
        if case.category == "event_ordering":
            tau = await self._event_ordering_tau(case, answer)
            score = (nugget_mean + (tau + 1.0) / 2.0) / 2.0
            metrics["kendall_tau_b"] = round(tau, 4)
        metrics["judge_label"] = "PASS" if score >= 0.5 else "FAIL"
        return round(score, 4), metrics
