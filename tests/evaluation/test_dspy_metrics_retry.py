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


@pytest.fixture(autouse=True)
def _isolate_judge_cache(monkeypatch, tmp_path):
    """Disable the persistent judge cache by default so retry-path tests exercise
    the judge on every call; cache-specific tests re-enable it explicitly."""
    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE", "0")
    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE_DIR", str(tmp_path / "jc"))


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


@pytest.mark.asyncio
async def test_judge_cache_pins_identical_triple(tmp_path, monkeypatch):
    """An identical (question, gold, answer) triple returns the SAME score every
    call — the judge is invoked once, later calls hit the deterministic cache."""
    from unittest.mock import AsyncMock

    from evaluation.scoring.dspy_metrics import DspyScores as _S

    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE", "1")
    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE_DIR", str(tmp_path / "jc"))
    with patch("evaluation.scoring.dspy_metrics._build_openrouter_lm", return_value=MagicMock()):
        scorer = DspyMetricScorer()
    # noisy judge: would return a different score each call if not cached
    calls = {"n": 0}
    async def _noisy(q, g, p):
        calls["n"] += 1
        return _S(semantic_f1=1.0 if calls["n"] == 1 else 0.5)
    scorer._score_once = AsyncMock(side_effect=_noisy)

    s1 = await scorer.score("q", "gold", "answer")
    s2 = await scorer.score("q", "gold", "answer")
    s3 = await scorer.score("q", "gold", "answer")
    assert s1.semantic_f1 == s2.semantic_f1 == s3.semantic_f1 == 1.0  # pinned to first
    assert calls["n"] == 1  # judge called exactly once

    # a different answer is a different key -> judge runs again
    await scorer.score("q", "gold", "different answer")
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_judge_cache_disabled_scores_every_call(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from evaluation.scoring.dspy_metrics import DspyScores as _S

    monkeypatch.setenv("GRAPHKNOWS_JUDGE_CACHE", "0")
    with patch("evaluation.scoring.dspy_metrics._build_openrouter_lm", return_value=MagicMock()):
        scorer = DspyMetricScorer()
    assert scorer._cache_dir is None
    scorer._score_once = AsyncMock(return_value=_S(semantic_f1=0.7))
    await scorer.score("q", "g", "a")
    await scorer.score("q", "g", "a")
    assert scorer._score_once.await_count == 2  # no cache -> judged each time
