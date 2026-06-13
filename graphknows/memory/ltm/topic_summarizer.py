"""DSPy topic summarization for LTM communities."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import dspy

from graphknows.config.llm import normalize_openrouter_model

log = logging.getLogger(__name__)


class TopicSummarySignature(dspy.Signature):
    """Summarize a graph community from member entities and evidence passages."""

    entities: list[str] = dspy.InputField(desc="Entity names in one graph community.")
    passages: str = dspy.InputField(
        desc="Sample of source chunks connected to entities in the same community."
    )
    title: str = dspy.OutputField(desc="Short human-readable topic title.")
    summary: str = dspy.OutputField(desc="Two or three sentence grounded topic summary.")


class TopicSummaryModule(dspy.Module):
    """Tiny DSPy module for topic title and summary generation."""

    def __init__(self) -> None:
        super().__init__()
        self.summarize = dspy.Predict(TopicSummarySignature)

    def forward(self, entities: list[str], passages: str) -> Any:
        return self.summarize(entities=entities, passages=passages)


class TopicSummarizer:
    """OpenRouter-backed DSPy topic summarizer.

    When no API key is provided, callers still get deterministic fallback
    labels and empty summaries. This keeps LTM flushes usable in offline
    or LLM-free extraction modes while making topic summarization LLM-backed
    whenever a key is configured.
    """

    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = (api_key or "").strip()
        self._model = normalize_openrouter_model(model)
        self._module = TopicSummaryModule()
        self._lm = None
        if self._api_key:
            self._lm = dspy.LM(
                model=self._model,
                api_key=self._api_key,
                api_base="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=1024,
                num_retries=0,
            )

    async def summarize(
        self,
        *,
        members: list[str],
        passages: str,
        fallback_title: str,
    ) -> tuple[str, str]:
        """Return ``(title, summary)`` for one community."""
        if self._lm is None:
            return fallback_title, ""

        def _summarize_sync() -> Any:
            with dspy.settings.context(lm=self._lm):
                return self._module(entities=members, passages=passages)

        try:
            prediction = await asyncio.to_thread(_summarize_sync)
        except Exception as exc:
            log.warning("DSPy topic summarization failed: %s", exc)
            return fallback_title, ""

        title = str(getattr(prediction, "title", "") or fallback_title).strip()
        summary = str(getattr(prediction, "summary", "") or "").strip()
        return title or fallback_title, summary
