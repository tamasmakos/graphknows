"""The typed recall hit and the one renderer every adapter shares.

These cover the seam that four adapters used to re-derive independently: the
shape of a hit, where its date comes from, and how a ranked list of hits
becomes the memory block an answerer reads.
"""

from __future__ import annotations

from datetime import datetime

from graphknows.models.hit import Hit, render_memories


class TestHitWhen:
    def test_when_parses_the_raw_locomo_timestamp(self) -> None:
        """``ts`` arrives free-form from the store, not as ISO."""
        hit = Hit(text="Hey Jon!", ts="4:04 pm on 20 January, 2023")

        assert hit.when == datetime(2023, 1, 20, 16, 4)

    def test_when_falls_back_to_a_date_inside_the_text(self) -> None:
        """A turn whose date is prefixed into the text still dates correctly.

        This is the arm where speaker and timestamp were flattened into the
        utterance, so the hit carries no ``ts`` of its own. Only the calendar
        date is recovered — the scan matches a date, not a time — which is all
        the timeline needs to sort on.
        """
        hit = Hit(text="1:56 pm on 8 May, 2023 | Caroline: Hey Mel!")

        assert hit.when == datetime(2023, 5, 8)

    def test_when_is_none_when_nothing_carries_a_date(self) -> None:
        assert Hit(text="no dates here at all").when is None

    def test_when_is_naive(self) -> None:
        """Mixed aware/naive datetimes raise on sort; the renderer sorts."""
        assert Hit(text="x", ts="2023-01-20T10:00:00+02:00").when.tzinfo is None


class TestHitRender:
    def test_stamped_hit_renders_dated_and_attributed(self) -> None:
        hit = Hit(text="Mochi is a ragdoll", speaker="Gina", ts="4:04 pm on 20 January, 2023")

        assert hit.render() == "(January 20, 2023) Gina: Mochi is a ragdoll"

    def test_no_clock_time_reaches_the_block(self) -> None:
        """An answerer told to never carry a clock time must not be shown one."""
        assert "4:04" not in Hit(text="x", ts="4:04 pm on 20 January, 2023").render()

    def test_unstamped_hit_passes_through_untouched(self) -> None:
        """Its text already carries its own date inline — prefixing doubles it."""
        text = "1:56 pm on 8 May, 2023 | Caroline: Hey Mel!"

        assert Hit(text=text).render() == text

    def test_stamped_hit_without_a_speaker_is_still_dated(self) -> None:
        assert Hit(text="a fact", ts="20 January, 2023").render() == "(January 20, 2023) a fact"


class TestRenderMemories:
    def test_orders_dated_memories_chronologically_not_by_rank(self) -> None:
        hits = [
            Hit(text="third", ts="3 March, 2023"),
            Hit(text="first", ts="1 January, 2023"),
            Hit(text="second", ts="2 February, 2023"),
        ]

        out = render_memories(hits)

        assert [b.split(") ")[1] for b in out.text.split("\n\n")] == ["first", "second", "third"]

    def test_undated_memories_follow_the_timeline_in_retrieval_order(self) -> None:
        hits = [
            Hit(text="undated one"),
            Hit(text="dated", ts="1 January, 2023"),
            Hit(text="undated two"),
        ]

        out = render_memories(hits)

        assert out.text.split("\n\n") == [
            "(January 01, 2023) dated",
            "undated one",
            "undated two",
        ]
        assert (out.dated, out.undated, out.shown) == (1, 2, 3)

    def test_duplicate_blocks_are_dropped(self) -> None:
        hits = [Hit(text="same", ts="1 January, 2023")] * 3

        assert render_memories(hits).shown == 1

    def test_budget_drops_whole_memories_in_relevance_order(self) -> None:
        """Never truncate mid-sentence: a partial memory helps nobody."""
        hits = [Hit(text="a" * 100), Hit(text="b" * 100)]

        out = render_memories(hits, max_chars=150)

        assert out.text == "a" * 100

    def test_a_skipped_memory_does_not_stop_the_scan(self) -> None:
        """A shorter, lower-ranked memory may still fit inside the budget."""
        hits = [Hit(text="a" * 100), Hit(text="b" * 500), Hit(text="c" * 20)]

        out = render_memories(hits, max_chars=150)

        assert out.text == "a" * 100 + "\n\n" + "c" * 20

    def test_empty_hits_render_to_an_empty_block(self) -> None:
        out = render_memories([])

        assert not out
        assert (out.text, out.shown) == ("", 0)
