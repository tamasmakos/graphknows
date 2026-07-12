"""DSPy judge: (question, gold, generated) → a 0..1 correctness score.

Only the *judge* uses DSPy — answer generation stays a plain chat completion
(:mod:`evaluation.common.llm`). A single ``dspy.Predict`` over the signature
below is fired with the same model we use everywhere (``LLM_MODEL`` / the
optional ``JUDGE_MODEL`` override) through OpenRouter, and called concurrently
via DSPy's native async path (``acall``).
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

from graphknows.config.llm import get_openrouter_api_key

log = logging.getLogger(__name__)


class JudgeSignature(dspy.Signature):
    """Grade whether the generated answer correctly answers the question, using the gold answer as ground truth.

    Be lenient about surface form: paraphrases, supersets that still contain the
    gold fact, and differently formatted dates that convey the same day are
    correct. Score 1.0 when the generated answer conveys the gold answer, 0.0 when
    it is wrong, missing, or a refusal (unless the gold itself is a refusal), and a
    value in between for a partially correct answer (e.g. some items of a list).
    """

    question: str = dspy.InputField(desc="the question asked")
    gold: str = dspy.InputField(desc="the reference (golden) answer")
    generated: str = dspy.InputField(desc="the answer to evaluate")
    reasoning: str = dspy.OutputField(desc="one sentence justifying the score")
    score: float = dspy.OutputField(desc="correctness from 0.0 (wrong) to 1.0 (fully correct)")


class DSPyJudge:
    """One ``dspy.Predict`` judge bound to the configured OpenRouter model."""

    def __init__(self, model: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> None:
        if not model:
            raise RuntimeError("No judge model configured. Set LLM_MODEL/JUDGE_MODEL in .env.")
        key = get_openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
        # litellm (DSPy's backend) needs the openrouter/ route prefix.
        lm_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        self._lm = dspy.LM(lm_model, api_key=key, temperature=temperature, max_tokens=max_tokens)
        dspy.configure(lm=self._lm)
        self._predict = dspy.Predict(JudgeSignature)

    async def judge(self, question: str, gold: str, generated: str) -> tuple[float, str]:
        """Return ``(score_in_0_1, reasoning)`` for one triple."""
        pred = await self._predict.acall(question=question, gold=gold, generated=generated)
        return _clamp01(getattr(pred, "score", 0.0)), str(getattr(pred, "reasoning", "")).strip()

    async def aclose(self) -> None:
        """No persistent connection to close; kept for ChatLLM API parity."""
        return None


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
