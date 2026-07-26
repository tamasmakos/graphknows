"""Two-stage DSPy judge: signature shape + stage1->stage2 wiring (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.common.dspy_judge import DSPyJudge, VerdictSignature


def test_verdict_signature_fields() -> None:
    """VerdictSignature carries both quantitative inputs and a graded output."""
    fields = set(VerdictSignature.model_fields)
    assert {"question", "gold", "generated", "semantic_f1", "token_f1"} <= fields
    assert {"reasoning", "score"} <= fields
    # semantic_f1 / token_f1 are evidence INPUTS; score is the OUTPUT verdict.
    assert {"semantic_f1", "token_f1"} <= set(VerdictSignature.input_fields)
    assert "score" in VerdictSignature.output_fields


class _FakePredict:
    """Stand-in for a dspy.Predict; records kwargs, returns a fixed score."""

    def __init__(self, score: float, reasoning: str = "r") -> None:
        self._score = score
        self._reasoning = reasoning
        self.kwargs: dict[str, Any] = {}

    async def acall(self, **kwargs: Any) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(score=self._score, reasoning=self._reasoning)


def test_judge_lm_is_built_with_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge LM without a timeout wedges the whole run, silently and forever.

    Observed: a conv-30 evaluation sat on one judge call for 2h49m with the
    process alive and the log frozen. litellm has no default socket timeout, so
    a dropped upstream connection never raises and never returns. The judge must
    route through build_lm, which is the single place that pins the timeout (and
    the OpenRouter api_base).
    """
    seen: dict[str, Any] = {}

    def _fake_lm(**kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    import dspy

    monkeypatch.setattr(dspy, "LM", _fake_lm)
    monkeypatch.setattr(dspy, "configure", lambda **_: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    DSPyJudge("meta-llama/llama-3.3-70b-instruct")

    assert seen.get("timeout"), "judge LM built without a timeout"
    assert seen["api_base"] == "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_judge_two_stage_wires_stage1_into_stage2() -> None:
    # Bypass __init__ (no API key / no dspy.configure): inject fake predictors.
    judge = DSPyJudge.__new__(DSPyJudge)
    judge._predict = _FakePredict(0.4)  # stage-1 SemanticF1
    judge._verdict_predict = _FakePredict(0.9)  # stage-2 final verdict

    score, metrics = await judge.judge_two_stage(
        "When did Caroline join?", "7 May 2023", "7 May 2023"
    )

    # Headline score is the STAGE-2 verdict, not the stage-1 SemanticF1.
    assert score == 0.9
    assert metrics["judge_score"] == 0.9
    assert metrics["semantic_f1"] == 0.4  # stage-1 preserved for divergence
    assert metrics["token_f1"] == 1.0  # exact lexical match on identical strings
    assert "judge_reasoning" in metrics

    # Stage-2 actually received the stage-1 score + token_f1 as evidence inputs.
    assert judge._verdict_predict.kwargs["semantic_f1"] == 0.4
    assert judge._verdict_predict.kwargs["token_f1"] == metrics["token_f1"]
