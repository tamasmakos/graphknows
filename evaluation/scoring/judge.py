"""LLM-as-judge scorer for LoCoMo — the metric modern leaderboards actually use.

token-F1 is lexical: it fails ``"dance"`` vs ``"by dancing"`` and ``"2"`` vs
``"two"`` even though they are correct. The published LoCoMo leaderboards score
*semantic* correctness with an LLM judge. This runs alongside token-F1 so every
case carries both a lexical and a semantic verdict.
"""

from __future__ import annotations

import asyncio
import logging
import os

import dspy

from graphknows.config.llm import _get_openrouter_api_key, normalize_openrouter_model

log = logging.getLogger(__name__)

class LoCoMoJudge(dspy.Signature):
    """You are a strict answer-correctness judge for a conversational memory benchmark.

    Mark 'yes' ONLY when the predicted answer is fully correct:
    - It covers ALL significant facts/items in the gold answer (no omissions).
    - Paraphrases and synonyms are fine ("dance" == "by dancing", "2" == "two").
    - Verbose answers that CONTAIN the full gold fact are fine.

    Mark 'no' when:
    - Any gold item or key fact is missing from the prediction.
    - The prediction adds contradicting information.
    - The prediction only partially overlaps (e.g. gold="psychology, counseling
      certification" and prediction="counseling, mental health" — 'counseling'
      overlaps but 'psychology' is missing and 'mental health' is invented → no).

    Respond strictly 'yes' or 'no'. No other output."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    correct: str = dspy.OutputField(desc="'yes' or 'no' — no other text")


class LLMJudge:
    """Per-case semantic correctness verdict via an OpenRouter LLM.

    Initialize once per run and reuse. ``judge`` is async (``asyncio.to_thread``)
    and rate-limit aware, mirroring :class:`~evaluation.generation.generator.AnswerGenerator`.
    """

    def __init__(
        self, model: str | None = None, api_key: str | None = None, samples: int = 3
    ) -> None:
        resolved_key = api_key or _get_openrouter_api_key()
        if not resolved_key:
            raise RuntimeError("LLMJudge requires an OpenRouter API key (OPENROUTER_API_KEY).")
        model_str = normalize_openrouter_model(
            model or os.environ.get("OPENROUTER_MODEL") or os.environ.get("LLM_MODEL") or ""
        )
        if not model_str:
            raise RuntimeError("No judge model configured. Set LLM_MODEL in .env.")
        # The judge is non-deterministic on borderline/partial answers even at
        # temperature 0 (reasoning models vary run-to-run). A single verdict gives
        # the headline accuracy a multi-point noise band on exactly the cases that
        # separate methods, so each case is judged ``samples`` times and decided by
        # majority vote (with early exit once one side is unbeatable).
        self._samples = max(1, samples)
        self._lm = dspy.LM(
            model=model_str,
            api_key=resolved_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=1024,  # reasoning models burn the budget on reasoning_content before {"correct":...}
            num_retries=0,
            cache=False,
        )
        self._predictor = dspy.Predict(LoCoMoJudge)
        log.info("LLMJudge ready with model %s (majority-of-%d)", model_str, self._samples)

    def _judge_sync(self, question: str, gold: str, predicted: str) -> bool:
        with dspy.settings.context(lm=self._lm):
            pred = self._predictor(question=question, gold_answer=gold, predicted_answer=predicted)
        return str(getattr(pred, "correct", "")).strip().lower().startswith("y")

    async def _judge_once(self, question: str, gold: str, predicted: str) -> bool:
        """One rate-limit-aware verdict. Parse failures score as incorrect."""
        for attempt in range(6):
            try:
                return await asyncio.to_thread(self._judge_sync, question, gold, predicted)
            except Exception as exc:
                msg = str(exc).lower()
                if "402" in msg or "credits" in msg or "payment" in msg:
                    raise RuntimeError(f"OpenRouter credit limit reached: {exc}") from exc
                if "429" in msg or "rate" in msg or "quota" in msg:
                    await asyncio.sleep(20 * (attempt + 1))
                elif "400" in msg or "bad request" in msg or "invalid_request" in msg:
                    await asyncio.sleep(4 * (attempt + 1))
                elif "parse" in msg or "empty or null response" in msg:
                    # Reasoning-model truncation (see AnswerGenerator) — deterministic
                    # at temp 0, so score as incorrect instead of crashing the run.
                    log.warning("Judge parse failure; scoring as incorrect: %s", question[:60])
                    return False
                else:
                    raise
        log.error("Judge failed after retries for: %s", question[:80])
        return False

    async def judge(self, question: str, gold: str, predicted: str) -> bool:
        """Return the majority verdict on whether *predicted* is correct for *gold*."""
        if not predicted or predicted.strip().lower().startswith("unknown"):
            return False
        needed = self._samples // 2 + 1
        yes = no = 0
        for _ in range(self._samples):
            if await self._judge_once(question, gold, predicted):
                yes += 1
            else:
                no += 1
            if yes >= needed or no >= needed:  # early exit: outcome already decided
                break
        return yes >= needed
