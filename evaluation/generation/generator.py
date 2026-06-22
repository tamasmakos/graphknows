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

_DEFAULT_MODEL = "moonshotai/kimi-k2"


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
        max_tokens: int = 256,
    ) -> None:
        resolved_key = api_key or _get_openrouter_api_key()
        if not resolved_key:
            raise RuntimeError(
                "AnswerGenerator requires an OpenRouter API key. "
                "Set OPENROUTER_API_KEY in the environment."
            )
        model_str = normalize_openrouter_model(
            model or os.environ.get("OPENROUTER_MODEL") or _DEFAULT_MODEL
        )
        self._lm = dspy.LM(
            model=model_str,
            api_key=resolved_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=max_tokens,
            num_retries=2,
            cache=False,  # DSPy/litellm caches by default; identical outputs would mask changes
        )
        self._predictor = dspy.Predict(LoCoMoAnswerGeneration)
        self._predictor.demos = demos()
        log.info("AnswerGenerator ready with model %s (%d demos)", model_str, len(demos()))

    def _generate_sync(self, question: str, context: str) -> str:
        with dspy.settings.context(lm=self._lm):
            pred = self._predictor(question=question, retrieved_context=context)
        return str(getattr(pred, "answer", None) or "Unknown").strip()

    async def generate(self, question: str, passages: list[RetrievedPassage]) -> str:
        """Generate an answer from retrieved passages (async-safe, rate-limit aware)."""
        context = build_context(passages)
        for attempt in range(5):
            try:
                return await asyncio.to_thread(self._generate_sync, question, context)
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "quota" in msg:
                    wait = 30 * (attempt + 1)
                    log.warning("Rate limit hit, waiting %ds (attempt %d/5)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    raise
        return "Unknown"
