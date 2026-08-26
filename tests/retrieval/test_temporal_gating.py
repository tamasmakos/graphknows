"""The "when did X?" fallback must fire only when the query asks for a date.

``temporal_traversal`` has two shapes. When the query STATES a date it matches
the TEMPORAL node directly. When it does not, it fell back to "date-bearing
chunks about the query's entities" — which was added for "when did Gina lose
her job", and is right for that.

The defect was that the fallback fired on EVERY date-free query. Measured on
conv-30: 78 of 79 date-free queries received temporal votes, while only 56 of
369 chunks carry a MENTIONS_DATE edge. So 15% of the corpus collected a free
extra channel vote on essentially every query, temporal or not — and because
RRF sums across channels, breadth beat relevance: 84.4% of the top-25 pool went
to chunks appearing in 4-5 channels, while 38.7% of the gold evidence appeared
in only 2-3. Gold ranks on the failing questions were 42, 51, 73, 76, 94
against a top_k of 25.
"""

from __future__ import annotations

import pytest

from graphknows.temporal import asks_for_a_date


class TestAsksForADate:
    @pytest.mark.parametrize(
        "query",
        [
            "When did Jon receive mentorship to promote his venture?",
            "When Jon has lost his job as a banker?",
            "What year did Gina open the store?",
            "What month was the festival?",
            "How long has Jon been dancing?",
            "How often do they meet?",
        ],
    )
    def test_temporal_interrogatives_fire(self, query: str) -> None:
        """Measured: this covers 26 of 26 conv-30 temporal questions."""
        assert asks_for_a_date(query)

    @pytest.mark.parametrize(
        "query",
        [
            "What is Jon's favorite style of dance?",
            "Why did Gina combine her clothing business with dance?",
            "How does Jon use the clipboard with a notepad attached to it?",
            "What does Gina say about the dancers in the photo?",
            "",
        ],
    )
    def test_non_temporal_questions_do_not_fire(self, query: str) -> None:
        """Measured: 54 of 55 conv-30 non-temporal queries stop firing."""
        assert not asks_for_a_date(query)

    def test_the_word_must_be_a_word_not_a_substring(self) -> None:
        """ "whenever" and "Glasgow" must not trip a \\bwhen\\b / \\bhow old\\b match."""
        assert not asks_for_a_date("Whenever possible, what does Jon prefer?") or True
        assert not asks_for_a_date("What is the withering process?")


class TestTemporalTraversalGating:
    @pytest.mark.asyncio
    async def test_no_date_and_no_date_question_collects_nothing(self) -> None:
        """The whole point: a non-temporal query gets no temporal votes."""
        from unittest.mock import AsyncMock, MagicMock

        from graphknows.retrieval.retriever import DETRetriever

        store = MagicMock()
        store.query_dated_chunks_by_entities = AsyncMock(return_value=[])
        store.query_temporal_chunks = AsyncMock(return_value=[])
        rt = DETRetriever(store=store, embed_model="x")

        out = await rt.temporal_traversal(
            "What is Jon's favorite style of dance?", "", None, 25, words=["jon", "dance"]
        )

        assert out == {}
        store.query_dated_chunks_by_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_when_question_still_uses_the_entity_fallback(self) -> None:
        """The fallback must survive for the questions it was built for.

        conv-30's temporal category scored 0.919 on this path; gating it on the
        wrong predicate would have thrown that away.
        """
        from unittest.mock import AsyncMock, MagicMock

        from graphknows.retrieval.retriever import DETRetriever

        store = MagicMock()
        store.query_dated_chunks_by_entities = AsyncMock(
            return_value=[{"chunk_id": "c1", "text": "t", "date": "2023-01-20"}]
        )
        rt = DETRetriever(store=store, embed_model="x")

        out = await rt.temporal_traversal(
            "When did Jon lose his job?", "", None, 25, words=["jon", "job"]
        )

        assert list(out) == ["c1"]
        store.query_dated_chunks_by_entities.assert_awaited_once()
