"""Adapter judge/answer behavior against a scripted fake LLM (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.llm import parse_json_object
from evaluation.locomo.adapter import LocomoAdapter
from evaluation.locomo.config import LocomoConfig

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.0}


class FakeLLM:
    """Returns scripted completions in order; records every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    async def generate(self, system: str, user: str, **_: Any) -> tuple[str, dict]:
        self.prompts.append((system, user))
        return self._replies.pop(0), dict(_USAGE)

    async def generate_json(self, system: str, user: str, **_: Any) -> tuple[dict, dict]:
        text, usage = await self.generate(system, user)
        return parse_json_object(text), usage


@pytest.fixture(autouse=True)
def _no_judge_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE", "0")


def _case(**overrides: Any) -> EvalCase:
    base: dict[str, Any] = {
        "case_id": "c1",
        "group_id": "g1",
        "question": "When did Caroline go to the support group?",
        "gold": "7 May 2023",
        "category": "temporal",
    }
    base.update(overrides)
    return EvalCase(**base)


def _unit() -> IngestUnit:
    return IngestUnit(group_id="g1", reference_date="8 May 2023")


def _passages() -> list[RetrievedPassage]:
    return [RetrievedPassage(text="[2023/05/08] Caroline: I went yesterday.")]


class TestLocomoAdapter:
    @pytest.mark.asyncio
    async def test_answer_extracts_after_marker(self) -> None:
        answerer = FakeLLM(["Stage 1 ...\nANSWER: 7 May 2023"])
        adapter = LocomoAdapter(LocomoConfig(), answerer, FakeLLM([]))
        answer, usage = await adapter.answer(_case(), _unit(), _passages())
        assert answer == "7 May 2023"
        assert usage["gen_total_tokens"] == 15
        assert "8 May 2023" in answerer.prompts[0][1]  # reference date injected

    @pytest.mark.asyncio
    async def test_judge_correct_and_wrong(self) -> None:
        judge = FakeLLM(
            ['{"reasoning": "same date", "label": "CORRECT"}',
             '{"reasoning": "different", "label": "WRONG"}']
        )
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), judge)
        score, metrics = await adapter.judge_case(_case(), "7 May 2023")
        assert score == 1.0 and metrics["judge_label"] == "CORRECT"
        score, _ = await adapter.judge_case(_case(), "12 Oct 2023")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_open_ended_gold_truncated_for_judge(self) -> None:
        judge = FakeLLM(['{"reasoning": "ok", "label": "CORRECT"}'])
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), judge)
        case = _case(category="open_ended", gold="painting; because she loves art")
        await adapter.judge_case(case, "painting")
        assert "because she loves art" not in judge.prompts[0][1]

    def test_lexical_metrics_present(self) -> None:
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), FakeLLM([]))
        metrics = adapter.lexical_metrics(_case(), "7 May 2023")
        assert metrics["locomo_f1"] == 1.0 and metrics["token_f1"] == 1.0
