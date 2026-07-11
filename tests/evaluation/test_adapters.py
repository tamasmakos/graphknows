"""Adapter judge/answer behavior against a scripted fake LLM (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.beam.adapter import BeamAdapter
from evaluation.beam.config import BeamConfig
from evaluation.common.datamodels import EvalCase, IngestUnit, RetrievedPassage
from evaluation.common.llm import parse_json_object
from evaluation.locomo.adapter import LocomoAdapter
from evaluation.locomo.config import LocomoConfig
from evaluation.longmemeval.adapter import LongMemEvalAdapter, parse_yes_no
from evaluation.longmemeval.config import LongMemEvalConfig

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


class TestLongMemEvalAdapter:
    @pytest.mark.asyncio
    async def test_answer_uses_question_date(self) -> None:
        answerer = FakeLLM(["<mem_thinking>scan</mem_thinking>\nANSWER: two trips"])
        adapter = LongMemEvalAdapter(LongMemEvalConfig(), answerer, FakeLLM([]))
        case = _case(question_date="2023/06/01 (Thu)")
        answer, _ = await adapter.answer(case, _unit(), _passages())
        assert answer == "two trips"
        assert "2023/06/01 (Thu)" in answerer.prompts[0][1]

    @pytest.mark.asyncio
    async def test_judge_yes_no(self) -> None:
        judge = FakeLLM(
            ["<judge_thinking>close enough</judge_thinking>\nyes",
             "<judge_thinking>missing the fact</judge_thinking>\nno"]
        )
        adapter = LongMemEvalAdapter(LongMemEvalConfig(), FakeLLM([]), judge)
        score, metrics = await adapter.judge_case(_case(), "two trips")
        assert score == 1.0 and metrics["judge_label"] == "yes"
        score, _ = await adapter.judge_case(_case(), "no idea")
        assert score == 0.0

    def test_parse_yes_no_variants(self) -> None:
        assert parse_yes_no("<judge_thinking>no no no</judge_thinking>\nYes.") is True
        assert parse_yes_no("verdict:\nno") is False
        assert parse_yes_no("The answer is yes overall") is True
        assert parse_yes_no("") is None


class TestBeamAdapter:
    @pytest.mark.asyncio
    async def test_nugget_mean(self) -> None:
        judge = FakeLLM(
            ['{"score": 1.0, "reason": "present"}', '{"score": 0.0, "reason": "missing"}']
        )
        adapter = BeamAdapter(BeamConfig(), FakeLLM([]), judge)
        case = _case(
            category="information_extraction",
            extras={"nuggets": ["mentions the guitar", "mentions the price"]},
        )
        score, metrics = await adapter.judge_case(case, "bought a guitar")
        assert score == 0.5
        assert metrics["nugget_scores"] == [1.0, 0.0]
        assert metrics["judge_label"] == "PASS"

    @pytest.mark.asyncio
    async def test_event_ordering_blends_tau(self) -> None:
        judge = FakeLLM(
            [
                '{"score": 1.0, "reason": "a"}',
                '{"score": 1.0, "reason": "b"}',
                '["event A", "event B"]',  # extraction
                '{"index": 0, "reason": "matches A"}',
                '{"index": 1, "reason": "matches B"}',
            ]
        )
        adapter = BeamAdapter(BeamConfig(), FakeLLM([]), judge)
        case = _case(
            category="event_ordering",
            extras={"nuggets": ["first: A happens", "then: B happens"]},
        )
        score, metrics = await adapter.judge_case(case, "A then B")
        assert metrics["kendall_tau_b"] == 1.0
        assert score == 1.0  # (nugget 1.0 + (tau+1)/2 = 1.0) / 2

    @pytest.mark.asyncio
    async def test_no_rubric_scores_zero(self) -> None:
        adapter = BeamAdapter(BeamConfig(), FakeLLM([]), FakeLLM([]))
        case = _case(gold="", extras={})
        score, metrics = await adapter.judge_case(case, "anything")
        assert score == 0.0 and metrics["judge_label"] == "NO_RUBRIC"
