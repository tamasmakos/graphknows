"""Retrieval strategy is selectable per call, independent of the ingest mode.

Before this, GRAPHKNOWS_MODE forked ingestion and retrieval together and the
retriever was memoised per process, so answering one query agentically and the
next deterministically meant a server restart -- and asking whether the
llm_assisted gains came from DSPy ingestion or from agentic retrieval was not
expressible at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphknows.retrieval import build_retriever
from graphknows.settings import GraphKnowsSettings, MemoryMode, RetrievalStrategy


def _settings(**kw: object) -> GraphKnowsSettings:
    return GraphKnowsSettings(GRAPHKNOWS_LLM_API_KEY="k", **kw)  # type: ignore[arg-type]


class TestBuildRetriever:
    def test_explicit_strategy_beats_the_mode_preset(self) -> None:
        s = _settings(GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert s.retrieval_strategy is RetrievalStrategy.det
        r = build_retriever(s, store=MagicMock(), strategy="agentic")
        assert type(r).__name__ == "AgenticRetriever"

    def test_explicit_det_over_an_llm_assisted_config(self) -> None:
        s = _settings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted)
        assert s.retrieval_strategy is RetrievalStrategy.agentic
        r = build_retriever(s, store=MagicMock(), strategy="det")
        assert type(r).__name__ == "DETRetriever"

    def test_empty_strategy_uses_the_configured_default(self) -> None:
        assert type(build_retriever(_settings(), store=MagicMock())).__name__ == "DETRetriever"
        s = _settings(GRAPHKNOWS_RETRIEVAL=RetrievalStrategy.agentic)
        assert type(build_retriever(s, store=MagicMock())).__name__ == "AgenticRetriever"

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid RetrievalStrategy"):
            build_retriever(_settings(), store=MagicMock(), strategy="telepathy")


class TestMemoryRetrieverCache:
    @pytest.mark.asyncio
    async def test_two_strategies_share_one_store(self) -> None:
        """A second strategy must not open a second connection."""
        from graphknows.memory import Memory

        mem = Memory(_settings(), namespace="")
        store = MagicMock()
        mem._retriever_stores = [store]  # pretend the first connect already happened

        det = await mem._get_retriever("det")
        agentic = await mem._get_retriever("agentic")

        assert type(det).__name__ == "DETRetriever"
        assert type(agentic).__name__ == "AgenticRetriever"
        assert mem._retriever_stores == [store], "a second store was opened"
        assert sorted(mem._retrievers) == ["agentic", "det"]

    @pytest.mark.asyncio
    async def test_same_strategy_is_memoised(self) -> None:
        from graphknows.memory import Memory

        mem = Memory(_settings(), namespace="")
        mem._retriever_stores = [MagicMock()]
        assert await mem._get_retriever("det") is await mem._get_retriever("det")

    @pytest.mark.asyncio
    async def test_default_and_explicit_share_a_cache_entry(self) -> None:
        """ "" and "det" resolve to the same key under llm_free, not two retrievers."""
        from graphknows.memory import Memory

        mem = Memory(_settings(GRAPHKNOWS_MODE=MemoryMode.llm_free), namespace="")
        mem._retriever_stores = [MagicMock()]
        assert await mem._get_retriever("") is await mem._get_retriever("det")
        assert list(mem._retrievers) == ["det"]
