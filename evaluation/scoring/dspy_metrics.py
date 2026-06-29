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
import logging
import os
from dataclasses import dataclass

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


def _build_openrouter_lm(model: str | None, api_key: str | None) -> "dspy.LM":
    """Construct an OpenRouter-backed ``dspy.LM``."""
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
        temperature=0.0,
        max_tokens=2048,
        num_retries=0,
        cache=False,
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
        lm: "dspy.LM | None" = None,
        threshold: float = 0.66,
        decompositional: bool = False,
    ) -> None:
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is not installed. Install it to use DspyMetricScorer."
            )
        self._lm = lm or _build_openrouter_lm(model, api_key)
        self._semantic_f1 = _SemanticF1(threshold=threshold, decompositional=decompositional)
        log.info("DspyMetricScorer ready (SemanticF1)")

    def _score_sync(self, question: str, gold: str, predicted: str) -> DspyScores:
        example = dspy.Example(question=question, response=gold).with_inputs("question")
        pred = dspy.Prediction(response=predicted or "Unknown")
        with dspy.settings.context(lm=self._lm):
            sf1 = float(self._semantic_f1(example, pred).score)
        return DspyScores(semantic_f1=sf1)

    async def score(self, question: str, gold: str, predicted: str) -> DspyScores:
        """Return the SemanticF1 score for one case (rate-limit aware)."""
        for attempt in range(6):
            try:
                return await asyncio.to_thread(self._score_sync, question, gold, predicted)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "402" in msg or "credits" in msg or "payment" in msg:
                    raise RuntimeError(f"OpenRouter credit limit reached: {exc}") from exc
                if "429" in msg or "rate" in msg or "quota" in msg:
                    await asyncio.sleep(20 * (attempt + 1))
                elif "400" in msg or "bad request" in msg or "invalid_request" in msg:
                    await asyncio.sleep(4 * (attempt + 1))
                elif "parse" in msg or "empty or null response" in msg:
                    log.warning("SemanticF1 parse failure; scoring 0.0: %s", question[:60])
                    return DspyScores(semantic_f1=0.0)
                else:
                    raise
        log.error("SemanticF1 scoring failed after retries for: %s", question[:80])
        return DspyScores(semantic_f1=0.0)
