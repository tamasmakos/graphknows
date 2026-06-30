"""Regression tests for the SemanticF1 judge truncation/parse-failure handling.

The judge LM (a reasoning model at temp=0) can enter a deterministic repetition
loop that truncates its own output; DSPy surfaces that as a parse failure. The
scorer must NOT record a correct answer as 0.0 in that case — it must retry once
with a small-temperature judge to break the loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evaluation.scoring.dspy_metrics import DspyMetricScorer, DspyScores


async def _direct(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@pytest.mark.asyncio
async def test_parse_failure_retries_with_temp_judge_and_does_not_score_zero():
    primary, retry = MagicMock(name="primary_lm"), MagicMock(name="retry_lm")
    with patch(
        "evaluation.scoring.dspy_metrics._build_openrouter_lm", side_effect=[primary, retry]
    ):
        scorer = DspyMetricScorer()
    assert scorer._lm is primary and scorer._lm_retry is retry

    seen_lms = []

    def fake_score_sync(question, gold, predicted, lm=None):
        seen_lms.append(lm)
        if lm is retry:
            return DspyScores(semantic_f1=0.83)
        # temp=0 primary judge truncated -> DSPy parse failure
        raise RuntimeError("failed to parse structured output (truncated)")

    with (
        patch.object(scorer, "_score_sync", side_effect=fake_score_sync),
        patch("asyncio.to_thread", side_effect=_direct),
    ):
        result = await scorer.score("q", "gold", "a correct answer")

    assert result.semantic_f1 == 0.83, "correct answer must not be scored 0.0 on truncation"
    assert seen_lms[0] is primary, "first attempt uses the temp=0 judge"
    assert seen_lms[1] is retry, "parse failure retries with the small-temperature judge"


@pytest.mark.asyncio
async def test_parse_failure_scores_zero_only_after_temp_retry_also_fails():
    primary, retry = MagicMock(name="primary_lm"), MagicMock(name="retry_lm")
    with patch(
        "evaluation.scoring.dspy_metrics._build_openrouter_lm", side_effect=[primary, retry]
    ):
        scorer = DspyMetricScorer()

    def always_parse_fail(question, gold, predicted, lm=None):
        raise RuntimeError("failed to parse structured output")

    with (
        patch.object(scorer, "_score_sync", side_effect=always_parse_fail),
        patch("asyncio.to_thread", side_effect=_direct),
    ):
        result = await scorer.score("q", "gold", "pred")

    # Only falls back to 0.0 once BOTH the temp=0 and the temp>0 judge fail to parse.
    assert result.semantic_f1 == 0.0
