"""TopicChannel routes a matched topic to its member chunks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.channels.base import ChannelContext
from graphknows.channels.topics import TopicChannel


def _ctx(session_id: str = "s1") -> ChannelContext:
    return ChannelContext(
        query="what did they plan",
        session_id=session_id,
        scope="both",
        emb=[0.1] * 8,
        emb2=None,
        words=[],
        words_graph=[],
        state=None,
    )


def _rt(rows: list[dict], chunk_total: int = 20, summaries: list[dict] | None = None) -> MagicMock:
    rt = MagicMock()
    rt.store.search_topics_with_chunks = AsyncMock(return_value=rows)
    rt.store.search_topics = AsyncMock(return_value=summaries or [])
    rt.store.query = AsyncMock(return_value=[{"c": chunk_total}])
    return rt


class TestRouting:
    @pytest.mark.asyncio
    async def test_no_topics_yields_nothing(self) -> None:
        assert await TopicChannel().collect(_ctx(), _rt([]), 5) == {}

    @pytest.mark.asyncio
    async def test_member_chunks_become_candidates(self) -> None:
        """The point of the change: topics route to real evidence, not just a summary."""
        rows = [
            {"chunk_id": "a", "topic_id": "t1", "score": 0.2, "keywords": [], "coherence": 0.8},
            {"chunk_id": "b", "topic_id": "t1", "score": 0.9, "keywords": [], "coherence": 0.8},
        ]
        out = await TopicChannel().collect(_ctx(), _rt(rows), 5)
        assert set(out) == {"a", "b"}
        assert out["b"][0] > out["a"][0]
        assert out["b"][1]["sources"] == {"topic"}

    @pytest.mark.asyncio
    async def test_scores_accumulate_across_topics(self) -> None:
        rows = [
            {"chunk_id": "a", "topic_id": "t1", "score": 0.4, "keywords": [], "coherence": 0.9},
            {"chunk_id": "a", "topic_id": "t2", "score": 0.4, "keywords": [], "coherence": 0.9},
            {"chunk_id": "b", "topic_id": "t2", "score": 0.7, "keywords": [], "coherence": 0.9},
        ]
        out = await TopicChannel().collect(_ctx(), _rt(rows), 5)
        # a appears in two matched topics (0.8 total) and outranks b (0.7).
        assert out["a"][0] > out["b"][0]


class TestSaturation:
    @pytest.mark.asyncio
    async def test_topic_covering_most_of_the_session_is_dropped(self) -> None:
        """A topic spanning the whole conversation is not a theme."""
        rows = [
            {"chunk_id": f"c{i}", "topic_id": "broad", "score": 0.5, "keywords": [],
             "coherence": 0.9}
            for i in range(9)
        ]
        assert await TopicChannel().collect(_ctx(), _rt(rows, chunk_total=10), 5) == {}

    @pytest.mark.asyncio
    async def test_narrow_topic_survives_a_saturated_one(self) -> None:
        rows = [
            *[
                {"chunk_id": f"c{i}", "topic_id": "broad", "score": 0.5, "keywords": [],
                 "coherence": 0.9}
                for i in range(9)
            ],
            {"chunk_id": "z", "topic_id": "narrow", "score": 0.8, "keywords": [],
             "coherence": 0.9},
        ]
        out = await TopicChannel().collect(_ctx(), _rt(rows, chunk_total=10), 5)
        assert set(out) == {"z"}

    @pytest.mark.asyncio
    async def test_without_a_session_nothing_is_dropped(self) -> None:
        rows = [
            {"chunk_id": f"c{i}", "topic_id": "broad", "score": 0.5, "keywords": [],
             "coherence": 0.9}
            for i in range(9)
        ]
        out = await TopicChannel().collect(_ctx(session_id=""), _rt(rows, chunk_total=10), 9)
        assert len(out) == 9


class TestSummaryPassage:
    @pytest.mark.asyncio
    async def test_summary_is_injected_with_keywords(self) -> None:
        summaries = [{"topic_id": "t1", "summary": "Trips and travel.",
                      "keywords": ["kayaking", "river"]}]
        out = await TopicChannel().collect(_ctx(), _rt([], summaries=summaries), 5)
        assert "topic:t1" in out
        text = out["topic:t1"][1]["text"]
        assert "Trips and travel." in text
        assert "kayaking, river" in text
        assert out["topic:t1"][1]["metadata"]["is_topic_summary"] is True

    @pytest.mark.asyncio
    async def test_empty_summary_is_not_injected(self) -> None:
        summaries = [{"topic_id": "t1", "summary": "  ", "keywords": []}]
        assert await TopicChannel().collect(_ctx(), _rt([], summaries=summaries), 5) == {}

    @pytest.mark.asyncio
    async def test_summary_survives_when_all_topics_are_saturated(self) -> None:
        """Saturation kills the routed chunks, not the abstractive passage."""
        rows = [
            {"chunk_id": f"c{i}", "topic_id": "broad", "score": 0.5, "keywords": [],
             "coherence": 0.9}
            for i in range(9)
        ]
        summaries = [{"topic_id": "broad", "summary": "The whole conversation.",
                      "keywords": []}]
        out = await TopicChannel().collect(
            _ctx(), _rt(rows, chunk_total=10, summaries=summaries), 5
        )
        assert set(out) == {"topic:broad"}
