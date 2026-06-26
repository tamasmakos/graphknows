"""DSPy answer generator for paper-comparable LoCoMo evaluation.

::

    generator = AnswerGenerator()           # reads OPENROUTER_API_KEY from env
    answer = await generator.generate(question, passages)
    # then score token F1(answer, gold)
"""

from __future__ import annotations

import asyncio
import logging
import os

import dspy

from evaluation.datamodels import RetrievedPassage
from evaluation.generation.signatures import (
    LoCoMoAnswerGeneration,
    build_context,
    demos,
)
from graphknows.config.llm import _get_openrouter_api_key, normalize_openrouter_model

log = logging.getLogger(__name__)

class AnswerGenerator:
    """Calls ``dspy.Predict(LoCoMoAnswerGeneration)`` for each (question, passages) pair.

    Initialize once per evaluation run and reuse across questions. The DSPy call
    is synchronous internally; :meth:`generate` is async via ``asyncio.to_thread``
    to avoid blocking the eval event loop.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 8192,  # base budget; reasoning models need >> the answer length
    ) -> None:
        resolved_key = api_key or _get_openrouter_api_key()
        if not resolved_key:
            raise RuntimeError(
                "AnswerGenerator requires an OpenRouter API key. "
                "Set OPENROUTER_API_KEY in the environment."
            )
        model_str = normalize_openrouter_model(
            model or os.environ.get("OPENROUTER_MODEL") or os.environ.get("LLM_MODEL") or ""
        )
        if not model_str:
            raise RuntimeError(
                "No generation model configured. Set LLM_MODEL in .env."
            )
        self._model = model_str
        self._api_key = resolved_key
        self._max_tokens = max_tokens
        # One LM per token budget, built lazily so generate() can escalate the
        # budget on a reasoning-model truncation (see below).
        self._lms: dict[int, dspy.LM] = {}
        # ChainOfThought: capture the reasoning, then emit the complete answer.
        # On a native reasoning model (e.g. deepseek-v4-flash) the model also
        # produces a reasoning_content channel; both share the token budget, which
        # is why the budget must be generous and escalate on truncation.
        self._predictor = dspy.ChainOfThought(LoCoMoAnswerGeneration)
        self._predictor.demos = demos()
        log.info(
            "AnswerGenerator ready with model %s (%d demos, CoT, max_tokens=%d)",
            model_str, len(demos()), max_tokens,
        )

    def _lm(self, max_tokens: int) -> dspy.LM:
        lm = self._lms.get(max_tokens)
        if lm is None:
            lm = dspy.LM(
                model=self._model,
                api_key=self._api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=max_tokens,
                num_retries=0,  # retries handled in generate() so we can re-route via OpenRouter
                cache=False,  # DSPy/litellm caches by default; identical outputs would mask changes
            )
            self._lms[max_tokens] = lm
        return lm

    def _generate_sync(self, question: str, context: str, max_tokens: int) -> tuple[str, dict]:
        lm = self._lm(max_tokens)
        history_len_before = len(lm.history)
        with dspy.settings.context(lm=lm):
            pred = self._predictor(question=question, retrieved_context=context)
        answer = str(getattr(pred, "answer", None) or "Unknown").strip()
        usage: dict = {}
        if len(lm.history) > history_len_before:
            entry = lm.history[-1]
            raw = entry.get("usage", {})
            usage = {
                "gen_prompt_tokens": int(raw.get("prompt_tokens", 0)),
                "gen_completion_tokens": int(raw.get("completion_tokens", 0)),
                "gen_total_tokens": int(raw.get("total_tokens", 0)),
                "gen_cost_usd": float(entry.get("cost") or 0.0),
            }
        return answer, usage

    async def generate(self, question: str, passages: list[RetrievedPassage]) -> tuple[str, dict]:
        """Generate an answer and return ``(answer, usage)`` with OpenRouter token/cost data.

        A reasoning model can spend the whole token budget on reasoning_content and
        return an empty structured body (DSPy raises a parse error). That is a
        truncation, NOT an abstention — so instead of returning "Unknown" we
        escalate the token budget once and retry; only a truncation that survives
        the larger budget falls back to "Unknown".
        """
        context = build_context(passages)
        for budget in (self._max_tokens, self._max_tokens * 4):
            for attempt in range(6):
                try:
                    return await asyncio.to_thread(self._generate_sync, question, context, budget)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "402" in msg or "credits" in msg or "payment" in msg:
                        raise RuntimeError(f"OpenRouter credit limit reached — add credits and restart: {exc}") from exc
                    elif "429" in msg or "rate" in msg or "quota" in msg:
                        wait = 30 * (attempt + 1)
                        log.warning("Rate limit hit, waiting %ds (attempt %d/6)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                    elif "400" in msg or "bad request" in msg or "invalid_request" in msg:
                        # OpenRouter may route to a provider (e.g. Novita) that rejects
                        # response_format. Each retry is a fresh OpenRouter request that
                        # may land on a different provider (e.g. Moonshot AI) that works.
                        wait = 5 * (attempt + 1)
                        log.warning("Provider 400 on attempt %d/6, retrying in %ds: %s", attempt + 1, wait, exc)
                        await asyncio.sleep(wait)
                    elif "parse" in msg or "empty or null response" in msg:
                        # Truncation: stop retrying at this budget and escalate.
                        log.warning(
                            "Truncation at max_tokens=%d (reasoning overflow); escalating: %s",
                            budget, question[:60],
                        )
                        break
                    else:
                        raise
        log.error("Generation failed at every budget; abstaining: %s", question[:80])
        return "Unknown", {}
