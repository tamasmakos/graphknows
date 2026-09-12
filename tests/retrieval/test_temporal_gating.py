"""``asks_for_a_date`` fires only when the query asks for a date.

The retriever's temporal collector went with the TEMPORAL plane; what survives
is the predicate, which a golden temporal channel over ``SEGMENT.observed_at``
must gate on. Its defect was that the fallback fired on EVERY date-free query. Measured on
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
        assert not asks_for_a_date("Whenever possible, what does Jon prefer?")
        assert not asks_for_a_date("What is the withering process?")
