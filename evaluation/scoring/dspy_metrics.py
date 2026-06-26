"""DSPy semantic scorers for LoCoMo: SemanticF1 + CompleteAndGrounded.

These wire the two reference DSPy evaluation metrics into the graphknows eval so
every case carries, alongside lexical token-F1 and the binary LLM judge, two
graded semantic signals:

* ``dspy.evaluate.SemanticF1``        — https://dspy.ai/api/evaluation/SemanticF1/
      LLM-estimated precision/recall of the *answer* vs the gold answer, combined
      into an F1. Captures paraphrase/synonym correctness that token-F1 misses.

* ``dspy.evaluate.CompleteAndGrounded`` — https://dspy.ai/api/evaluation/CompleteAndGrounded/
      F1 of two sub-scores:
        - completeness: does the answer cover the gold answer's key ideas?
        - groundedness: is the answer supported by the *retrieved context*?
      Groundedness is the key extra signal for this investigation: a low
      groundedness with a non-empty answer flags hallucination; a low
      completeness with high groundedness flags missing-evidence retrieval.

Both metrics call an LLM (each is a ``ChainOfThought`` module), so this module
needs ``OPENROUTER_API_KEY`` and a model exactly like
:mod:`evaluation.scoring.judge` and :mod:`evaluation.generation.generator`.

The LM is injectable (``lm=``) so the wiring can be unit-tested offline with a
stub LM (see ``evaluation/scripts/regrade_dspy_metrics.py --selftest``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import dspy
from dspy.evaluate import CompleteAndGrounded, SemanticF1

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DspyScores:
    """Both semantic scores for one case (NaN-free; 0.0 on failure)."""

    semantic_f1: float
    complete_and_grounded: float
    # The two CompleteAndGrounded sub-scores are also surfaced because they are
    # individually diagnostic (completeness ≈ retrieval/answer coverage,
    # groundedness ≈ hallucination check).
    completeness: float
    groundedness: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "semantic_f1": round(self.semantic_f1, 4),
            "complete_and_grounded": round(self.complete_and_grounded, 4),
            "completeness": round(self.completeness, 4),
            "groundedness": round(self.groundedness, 4),
        }


def _build_openrouter_lm(model: str | None, api_key: str | None) -> dspy.LM:
    """Construct an OpenRouter-backed ``dspy.LM`` mirroring the judge/generator."""
    # Imported lazily so the scorer wiring can be unit-tested with a stub LM on a
    # Python/host that cannot import the full graphknows package.
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
        max_tokens=2048,  # headroom for CoT reasoning + reasoning_content models
        num_retries=0,
        cache=False,
    )


class DspyMetricScorer:
    """Scores (question, gold, answer, context) with SemanticF1 + CompleteAndGrounded.

    Initialize once per run and reuse. ``score`` is async (``asyncio.to_thread``)
    and rate-limit aware, matching the other graphknows scorers. On any LM error
    that survives retries the affected score is returned as 0.0 so a single bad
    case never crashes the run — but the failure is logged so silent degradation
    is visible.

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
        self._lm = lm or _build_openrouter_lm(model, api_key)
        self._semantic_f1 = SemanticF1(threshold=threshold, decompositional=decompositional)
        self._complete_grounded = CompleteAndGrounded(threshold=threshold)
        log.info("DspyMetricScorer ready (SemanticF1 + CompleteAndGrounded)")

    # ------------------------------------------------------------------
    def _score_sync(self, question: str, gold: str, predicted: str, context: str) -> DspyScores:
        example = dspy.Example(question=question, response=gold).with_inputs("question")
        pred = dspy.Prediction(response=predicted or "Unknown", context=context or "")
        with dspy.settings.context(lm=self._lm):
            # SemanticF1: answer-vs-gold semantic F1.
            sf1 = float(self._semantic_f1(example, pred).score)
            # CompleteAndGrounded: re-run the two sub-modules directly so the
            # individual completeness/groundedness sub-scores can be reported,
            # then combine with the library's own f1.
            comp = self._complete_grounded.completeness_module(
                question=question, ground_truth=gold, system_response=pred.response
            )
            grnd = self._complete_grounded.groundedness_module(
                question=question, retrieved_context=pred.context, system_response=pred.response
            )
            from dspy.evaluate.auto_evaluation import f1_score

            completeness = float(comp.completeness)
            groundedness = float(grnd.groundedness)
            cg = float(f1_score(groundedness, completeness))
        return DspyScores(
            semantic_f1=sf1,
            complete_and_grounded=cg,
            completeness=completeness,
            groundedness=groundedness,
        )

    async def score(
        self, question: str, gold: str, predicted: str, context: str
    ) -> DspyScores:
        """Return both DSPy metric scores for one case (rate-limit aware)."""
        for attempt in range(6):
            try:
                return await asyncio.to_thread(
                    self._score_sync, question, gold, predicted, context
                )
            except Exception as exc:  # noqa: BLE001 - we classify and re-raise hard errors
                msg = str(exc).lower()
                if "402" in msg or "credits" in msg or "payment" in msg:
                    raise RuntimeError(f"OpenRouter credit limit reached: {exc}") from exc
                if "429" in msg or "rate" in msg or "quota" in msg:
                    await asyncio.sleep(20 * (attempt + 1))
                elif "400" in msg or "bad request" in msg or "invalid_request" in msg:
                    await asyncio.sleep(4 * (attempt + 1))
                elif "parse" in msg or "empty or null response" in msg:
                    log.warning("DSPy metric parse failure; scoring 0.0: %s", question[:60])
                    return DspyScores(0.0, 0.0, 0.0, 0.0)
                else:
                    raise
        log.error("DSPy metric scoring failed after retries for: %s", question[:80])
        return DspyScores(0.0, 0.0, 0.0, 0.0)
