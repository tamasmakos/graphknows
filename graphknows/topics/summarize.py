"""Topic summarisation — the one LLM step in the TOPIC layer.

Only used when ``GRAPHKNOWS_TOPICS=llm``. The extractive mode needs nothing
from here: it leads with the member chunk nearest the community centroid, which
is why the topic layer works at all in ``llm_free``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_MAX_PASSAGE_CHARS = 600
_PROMPT = (
    "These conversation excerpts all concern: {members}.\n\n"
    "{passages}\n\n"
    "Write ONE sentence naming what connects them. State only what the "
    "excerpts say. No preamble."
)


class TopicSummarizer:
    """Summarises a community's passages into a one-line topic description."""

    def __init__(self, model: str, api_key: str = "", api_base: str = "") -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._lm: Any = None

    def _get_lm(self) -> Any:
        if self._lm is None:
            from graphknows.llm import build_lm

            self._lm = build_lm(
                self._model,
                api_key=self._api_key,
                api_base=self._api_base,
                temperature=0.0,
                max_tokens=256,
                num_retries=1,
                timeout=120,  # hung calls wedged the whole flush before
            )
        return self._lm

    async def summarize(
        self, members: list[str], passages: list[str], fallback_title: str = ""
    ) -> str:
        """Return a one-sentence topic summary, or ``fallback_title`` on failure.

        Never raises: topic summarisation is the least important step in a
        flush, so a provider outage degrades the topic rather than the flush.
        """
        if not passages:
            return fallback_title
        prompt = _PROMPT.format(
            members=", ".join(members) or "these entities",
            passages="\n---\n".join(p[:_MAX_PASSAGE_CHARS] for p in passages),
        )
        try:
            lm = self._get_lm()
            out = await asyncio.to_thread(lm, prompt)
        except Exception as exc:
            log.warning("topic summarization failed (%s); using title", exc)
            return fallback_title
        text = out[0] if isinstance(out, list) and out else out
        return str(text).strip() or fallback_title


__all__ = ["TopicSummarizer"]
