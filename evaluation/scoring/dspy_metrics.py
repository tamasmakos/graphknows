"""DSPy SemanticF1 scorer for LoCoMo.

This is the single headline metric for the graphknows eval: DSPy's SemanticF1,
an LLM-estimated precision/recall of the predicted answer vs the gold answer,
combined into an F1. It captures paraphrase/synonym correctness that lexical
token-F1 misses.

Reference: https://dspy.ai/api/evaluation/SemanticF1/

The scorer needs ``OPENROUTER_API_KEY`` and ``LLM_MODEL`` from ``.env``.
The LM is injectable (``lm=``) for offline unit-testing with a stub LM.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median

try:
    import dspy
    from dspy.evaluate import SemanticF1 as _SemanticF1
    _DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DSPY_AVAILABLE = False
    dspy = None  # type: ignore[assignment]
    _SemanticF1 = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DspyScores:
    """SemanticF1 score for one case (0.0 on failure)."""

    semantic_f1: float

    def as_metrics(self) -> dict[str, float]:
        return {"semantic_f1": round(self.semantic_f1, 4)}


def _build_openrouter_lm(
    model: str | None,
    api_key: str | None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dspy.LM:
    """Construct an OpenRouter-backed ``dspy.LM`` for grading."""
    from graphknows.config.llm import _get_openrouter_api_key, normalize_openrouter_model

    resolved_key = api_key or _get_openrouter_api_key()
    if not resolved_key:
        raise RuntimeError(
            "DspyMetricScorer requires an OpenRouter API key (OPENROUTER_API_KEY)."
        )
    model_str = normalize_openrouter_model(
        model or os.environ.get("OPENROUTER_MODEL") or os.environ.get("LLM_MODEL") or ""
    )
    if not model_str:
        raise RuntimeError("No grading model configured. Set LLM_MODEL in .env.")
    return dspy.LM(
        model=model_str,
        api_key=resolved_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=0,
        cache=False,
        timeout=120,  # per-request; without it litellm defaults to 6000s → a hung call wedges the run
    )


class DspyMetricScorer:
    """Scores (question, gold, answer) with DSPy SemanticF1 — the headline metric.

    Initialize once per run and reuse. ``score`` is async (``asyncio.to_thread``)
    and rate-limit aware. On any LM error that survives retries the score is
    returned as 0.0 — a single bad case never crashes the run, but the failure is
    logged so silent degradation is visible.

    The ``lm`` argument is for testing: pass a stub LM to exercise the wiring with
    no network. In production leave it ``None`` and configure via env.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        lm: dspy.LM | None = None,
        threshold: float = 0.66,
        decompositional: bool = False,
    ) -> None:
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is not installed. Install it to use DspyMetricScorer."
            )
        self._lm = lm or _build_openrouter_lm(model, api_key)
        # Retry judge, used ONLY when the temp=0 judge truncates its own reasoning in
        # a deterministic repetition loop. DSPy surfaces that truncation as a parse
        # failure, which the scorer would otherwise record as 0.0 — mis-scoring a
        # CORRECT answer and injecting ~5-8 pts of spurious run-to-run noise on
        # long-answer conversations. A small temperature breaks the loop and the
        # larger budget covers genuinely long reasoning; 97% of calls never hit it,
        # so the metric stays temp=0-deterministic in the common case.
        self._lm_retry = (
            None if lm else _build_openrouter_lm(model, api_key, temperature=0.4, max_tokens=8192)
        )
        self._semantic_f1 = _SemanticF1(threshold=threshold, decompositional=decompositional)
        self._threshold = threshold
        self._decompositional = decompositional
        # Deterministic judge cache: SemanticF1 is a non-deterministic LLM judge
        # (~25% of triples vary run-to-run, up to ~0.3 on a single question). A
        # persistent (question, gold, answer) -> score cache makes an identical
        # triple always score identically — across reruns and A/B arms — so an
        # unchanged answer contributes exactly zero to a measured delta. Disable
        # with GRAPHKNOWS_JUDGE_CACHE=0. GRAPHKNOWS_JUDGE_SAMPLES>1 scores a
        # cache-miss N times and stores the median to also cut per-question noise.
        self._samples = max(1, int(os.environ.get("GRAPHKNOWS_JUDGE_SAMPLES", "1")))
        self._cache_dir: Path | None = None
        if os.environ.get("GRAPHKNOWS_JUDGE_CACHE", "1") not in ("0", "false", "off", ""):
            self._cache_dir = Path(
                os.environ.get("GRAPHKNOWS_JUDGE_CACHE_DIR",
                               "evaluation/results/.judge_cache")
            )
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "DspyMetricScorer ready (SemanticF1, cache=%s, samples=%d)",
            bool(self._cache_dir), self._samples,
        )

    def _cache_key(self, question: str, gold: str, predicted: str) -> str:
        """Stable key over the exact judge inputs (+ config that changes scoring)."""
        h = hashlib.sha256()
        for part in (question, gold, predicted or "Unknown",
                     str(self._threshold), str(self._decompositional)):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cache_get(self, key: str) -> float | None:
        if self._cache_dir is None:
            return None
        p = self._cache_dir / key
        if not p.exists():
            return None
        try:
            return float(p.read_text())
        except (ValueError, OSError):
            return None

    def _cache_put(self, key: str, value: float) -> None:
        if self._cache_dir is None:
            return
        p = self._cache_dir / key
        tmp = p.with_name(key + ".tmp")
        try:
            tmp.write_text(str(value))
            tmp.replace(p)  # atomic within the dir
        except OSError as exc:
            log.debug("judge cache write failed (non-fatal): %s", exc)

    def _score_sync(
        self, question: str, gold: str, predicted: str, lm: dspy.LM | None = None
    ) -> DspyScores:
        example = dspy.Example(question=question, response=gold).with_inputs("question")
        pred = dspy.Prediction(response=predicted or "Unknown")
        with dspy.settings.context(lm=lm or self._lm):
            sf1 = float(self._semantic_f1(example, pred).score)
        return DspyScores(semantic_f1=sf1)

    async def score(self, question: str, gold: str, predicted: str) -> DspyScores:
        """Return the SemanticF1 score for one case.

        Deterministic via the (question, gold, answer) cache: an identical triple
        always returns the same score. On a cache miss the (noisy) judge is called
        ``GRAPHKNOWS_JUDGE_SAMPLES`` times and the median is cached.
        """
        key = self._cache_key(question, gold, predicted)
        cached = self._cache_get(key)
        if cached is not None:
            return DspyScores(semantic_f1=cached)
        samples = [
            (await self._score_once(question, gold, predicted)).semantic_f1
            for _ in range(self._samples)
        ]
        value = float(median(samples))
        self._cache_put(key, value)
        return DspyScores(semantic_f1=value)

    async def _score_once(self, question: str, gold: str, predicted: str) -> DspyScores:
        """One SemanticF1 judge call (rate-limit aware; uncached)."""
        retried_temp = False
        for attempt in range(6):
            try:
                lm = self._lm_retry if (retried_temp and self._lm_retry is not None) else self._lm
                return await asyncio.to_thread(self._score_sync, question, gold, predicted, lm)
            except Exception as exc:
                msg = str(exc).lower()
                if "402" in msg or "credits" in msg or "payment" in msg:
                    raise RuntimeError(f"OpenRouter credit limit reached: {exc}") from exc
                if "429" in msg or "rate" in msg or "quota" in msg:
                    await asyncio.sleep(20 * (attempt + 1))
                elif "400" in msg or "bad request" in msg or "invalid_request" in msg:
                    await asyncio.sleep(4 * (attempt + 1))
                elif "parse" in msg or "empty or null response" in msg:
                    # The judge truncated its own reasoning (deterministic repetition
                    # loop at temp=0) and DSPy couldn't parse the structured score.
                    # Retry ONCE with a small-temperature judge to break the loop —
                    # scoring a correct answer 0.0 here is the bug we are fixing.
                    if not retried_temp and self._lm_retry is not None:
                        log.warning(
                            "SemanticF1 truncation/parse failure; retrying with temp>0 "
                            "to break repetition: %s", question[:60],
                        )
                        retried_temp = True
                        continue
                    log.warning(
                        "SemanticF1 parse failure after temp retry; scoring 0.0: %s",
                        question[:60],
                    )
                    return DspyScores(semantic_f1=0.0)
                elif "timeout" in msg or "timed out" in msg or "connection" in msg:
                    # Transient: a hung/dropped request now fast-fails (timeout=120 on the LM).
                    # Retry instead of aborting the whole run.
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
        log.error("SemanticF1 scoring failed after retries for: %s", question[:80])
        return DspyScores(semantic_f1=0.0)
