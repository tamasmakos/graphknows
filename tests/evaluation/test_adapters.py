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


class FakeJudge:
    """Scripted DSPy-style judge: returns queued scores, records (q, gold, gen).

    LoCoMo's adapter uses the two-stage judge, so ``judge_two_stage`` returns the
    queued value as the STAGE-2 verdict (the headline score) plus a metrics dict.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls: list[tuple[str, str, str]] = []

    async def judge(self, question: str, gold: str, generated: str) -> tuple[float, str]:
        self.calls.append((question, gold, generated))
        return self._scores.pop(0), "reason"

    async def judge_two_stage(
        self, question: str, gold: str, generated: str
    ) -> tuple[float, dict[str, Any]]:
        self.calls.append((question, gold, generated))
        score = self._scores.pop(0)
        return score, {
            "semantic_f1": score,
            "token_f1": 0.0,
            "judge_score": score,
            "judge_reasoning": "reason",
        }


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
    async def test_answer_prompt_includes_facts_block_when_present(self) -> None:
        answerer = FakeLLM(["ANSWER: x"])
        adapter = LocomoAdapter(LocomoConfig(), answerer, FakeLLM([]))
        await adapter.answer(
            _case(), _unit(), _passages(), ["Jon visiting Rome", "Gina support Jon"]
        )
        prompt = answerer.prompts[0][1]
        assert "Known facts" in prompt
        assert "- Jon visiting Rome" in prompt
        assert "- Gina support Jon" in prompt

    @pytest.mark.asyncio
    async def test_answer_prompt_byte_identical_to_baseline_when_no_facts(self) -> None:
        # facts=[] and facts=None must both render the pre-D8 prompt verbatim.
        base = FakeLLM(["ANSWER: a"])
        empty = FakeLLM(["ANSWER: b"])
        none = FakeLLM(["ANSWER: c"])
        LocomoConfig()
        await LocomoAdapter(LocomoConfig(), base, FakeLLM([])).answer(_case(), _unit(), _passages())
        await LocomoAdapter(LocomoConfig(), empty, FakeLLM([])).answer(
            _case(), _unit(), _passages(), []
        )
        await LocomoAdapter(LocomoConfig(), none, FakeLLM([])).answer(
            _case(), _unit(), _passages(), None
        )
        assert base.prompts[0][1] == empty.prompts[0][1] == none.prompts[0][1]
        assert "Known facts" not in base.prompts[0][1]

    @pytest.mark.asyncio
    async def test_judge_scores_map_to_labels(self) -> None:
        judge = FakeJudge([1.0, 0.0, 0.5])
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), judge)
        score, metrics = await adapter.judge_case(_case(), "7 May 2023")
        assert score == 1.0 and metrics["judge_label"] == "CORRECT"
        # Stage-1 SemanticF1 is recorded alongside the stage-2 headline score.
        assert metrics["semantic_f1"] == 1.0
        score, _ = await adapter.judge_case(_case(), "12 Oct 2023")
        assert score == 0.0
        score, metrics = await adapter.judge_case(_case(), "partly right")
        assert score == 0.5 and metrics["judge_label"] == "CORRECT"  # >= 0.5 passes

    @pytest.mark.asyncio
    async def test_open_ended_gold_truncated_for_judge(self) -> None:
        judge = FakeJudge([1.0])
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), judge)
        case = _case(category="open_ended", gold="painting; because she loves art")
        await adapter.judge_case(case, "painting")
        # The judge receives only the gold answer, not the trailing commentary.
        assert judge.calls[0][1] == "painting"

    def test_lexical_metrics_present(self) -> None:
        adapter = LocomoAdapter(LocomoConfig(), FakeLLM([]), FakeLLM([]))
        metrics = adapter.lexical_metrics(_case(), "7 May 2023")
        assert metrics["locomo_f1"] == 1.0 and metrics["token_f1"] == 1.0
