"""DSPy answer generator for paper-comparable LoCoMo evaluation.

The official LoCoMo paper (arXiv 2402.17753) scores token F1 between a
model-generated answer and the gold answer — not between retrieved context
and gold. This module adds the generation step so our scores are comparable.

Usage::

    generator = AnswerGenerator()           # reads OPENROUTER_API_KEY from env
    answer = await generator.generate(question, recall_hits)
    # then score token F1(answer, gold)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import dspy

from graphknows.config.llm import _get_openrouter_api_key, normalize_openrouter_model

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "moonshotai/kimi-k2"
_MAX_CONTEXT_CHARS = 8_000


class LoCoMoAnswerGeneration(dspy.Signature):
    """Generate a concise answer to a question about a long-term conversation,
    based only on the retrieved context passages provided."""

    question: str = dspy.InputField(desc="The question to answer about the conversation")
    retrieved_context: str = dspy.InputField(
        desc="Relevant conversation passages retrieved from memory, concatenated"
    )
    answer: str = dspy.OutputField(
        desc="A concise, direct answer. If the context does not contain enough information, say 'Unknown'."
    )


class AnswerGenerator:
    """Calls dspy.Predict(LoCoMoAnswerGeneration) for each (question, passages) pair.

    Designed to be initialized once per evaluation run and reused across all
    questions. The DSPy call is synchronous internally; ``generate`` is async
    via ``asyncio.to_thread`` to avoid blocking the eval event loop.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
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
            max_tokens=256,
            num_retries=0,
        )
        self._predictor = dspy.Predict(LoCoMoAnswerGeneration)
        log.info("AnswerGenerator ready with model %s", model_str)

    def _generate_sync(self, question: str, context: str) -> str:
        with dspy.settings.context(lm=self._lm):
            pred = self._predictor(question=question, retrieved_context=context)
        return str(getattr(pred, "answer", None) or "Unknown").strip()

    async def generate(self, question: str, passages: list[dict[str, Any]]) -> str:
        """Generate an answer from retrieved passages (async-safe)."""
        context = "\n\n".join(str(p.get("text", "")) for p in passages if p.get("text"))[
            :_MAX_CONTEXT_CHARS
        ]
        return await asyncio.to_thread(self._generate_sync, question, context)
