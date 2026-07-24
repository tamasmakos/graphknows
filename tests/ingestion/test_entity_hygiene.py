"""Entity-name hygiene: speaker-prefix reduction, temporal shapes, relation gate.

Pins the fixes for the conv-26 pollution findings: 27 speaker-prefixed entities
carried 42.5% of REL edges, and date names minted via the unfiltered
relation-endpoint path became ENTITY nodes with community_ids.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.extraction.entities.extractor import (
    _build_relation,
    _dedupe_entities,
    _reduce_speaker_span,
)
from graphknows.ingestion.stm._entity_writer import (
    graph_entities,
    is_graph_entity_name,
)

SPEAKERS = frozenset({"caroline", "melanie"})


class TestReduceSpeakerSpan:
    def test_known_speaker_prefix_reduces_to_the_speaker(self) -> None:
        assert _reduce_speaker_span("Caroline: Thanks, Melanie", SPEAKERS) == "Caroline"

    def test_case_insensitive_speaker_match(self) -> None:
        assert _reduce_speaker_span("MELANIE: hey there", SPEAKERS) == "MELANIE"

    def test_unknown_prefix_is_untouched(self) -> None:
        """A colon alone never fires — only a known speaker does."""
        assert (
            _reduce_speaker_span("Charlotte's Web: A Review", SPEAKERS)
            == "Charlotte's Web: A Review"
        )

    def test_no_speakers_is_the_identity(self) -> None:
        """Non-dialogue text has no speaker set, so nothing changes."""
        name = "Caroline: Thanks, Melanie"
        assert _reduce_speaker_span(name, frozenset()) == name

    def test_plain_name_is_untouched(self) -> None:
        assert _reduce_speaker_span("Caroline", SPEAKERS) == "Caroline"


class TestMintPoints:
    def test_dedupe_reduces_speaker_glued_span(self) -> None:
        """The span merges with the plain speaker entity instead of forking it."""
        raw = [
            {"text": "Caroline: Thanks, Melanie", "label": "person", "score": 0.9},
            {"text": "Caroline", "label": "person", "score": 0.8},
        ]
        out = _dedupe_entities(raw, {"person": "PERSON"}, SPEAKERS)
        assert [e["name"] for e in out] == ["Caroline"]
        assert out[0]["score"] == 0.9  # highest-scoring span won the merge key

    def test_relation_endpoints_are_reduced(self) -> None:
        rel = _build_relation(
            "Caroline: Thank you so much",
            "attended",
            "pride event",
            0.9,
            "relex",
            [],
            speakers=SPEAKERS,
        )
        assert rel is not None
        assert rel["head"] == "Caroline"

    def test_reduction_can_collapse_a_self_relation(self) -> None:
        """(Caroline: hi, told, Caroline) becomes head==tail and is dropped."""
        rel = _build_relation(
            "Caroline: hi", "told", "Caroline", 0.9, "relex", [], speakers=SPEAKERS
        )
        assert rel is None


class TestTemporalShapes:
    @pytest.mark.parametrize(
        "name",
        [
            "8 May, 2023",  # comma-year (previous gap)
            "1:56 pm on 8 May, 2023",  # time + trailing date (previous gap)
            "last Saturday",  # relative, no digit (previous gap)
            "next week",
            "this weekend",
            "Saturday",  # bare weekday — not a person name
            "yesterday",
            "23 July 2023",
            "July 2023",
            "10:31 am",
            "2023",
        ],
    )
    def test_temporal_names_are_junk(self, name: str) -> None:
        assert not is_graph_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "May",  # person name — months stay digit-gated
            "June",
            "Charlotte",
            "Caroline",
            "county hospital",
            "Mayfield",  # month prefix inside a real name
            "Sunday Times",  # weekday branch is $-anchored
        ],
    )
    def test_real_names_survive(self, name: str) -> None:
        assert is_graph_entity_name(name)

    def test_type_gate_still_applies(self) -> None:
        assert not is_graph_entity_name("anything", "DATE")
        assert is_graph_entity_name("anything", "PERSON")

    def test_graph_entities_uses_the_same_predicate(self) -> None:
        ents = [
            {"name": "Caroline", "type": "PERSON"},
            {"name": "last Saturday", "type": ""},
            {"name": "8 May, 2023", "type": ""},
        ]
        assert [e["name"] for e in graph_entities(ents)] == ["Caroline"]


class TestStructuralJunk:
    """Census of two live graphs found stopwords, code, URLs, month-first dates,
    and 1-2 char fragments minted as ENTITY nodes. All are structural junk."""

    @pytest.mark.parametrize(
        "name",
        [
            "the", "and", "or", "to", "be", "These", "what", "of", "a",  # stopwords
            "x", "id", "io", "db",  # single / 2-char lowercase fragments
            "::", "...", "-",  # punctuation-only
            "http://localhost:5000", "www.example.com",  # URLs
            "Flask application: ```python from flask import Flask",  # code
            "config = {a: 1}", "arr[]", "<div>",  # markup / code
            "December 15, 2023",  # month-first date (previous gap)
            "November 1 - November 15, 2023",  # date range (previous gap)
            "January 16 - February 15, 2024",
            "the quick brown fox jumped over the lazy dog today",  # >8-word clause
        ],
    )
    def test_structural_junk_is_dropped(self, name: str) -> None:
        assert not is_graph_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "Flask", "SQLAlchemy", "Werkzeug", "OpenWeatherMap",  # tech entities
            "python", "iPhone",  # lowercase / mixed-case real entities
            "AI", "US", "UK",  # uppercase acronyms survive the 2-char drop
            "May", "June",  # month-name collisions with modal/function words
            "San Francisco", "New York City", "county hospital",  # multi-word names
        ],
    )
    def test_real_entities_survive(self, name: str) -> None:
        assert is_graph_entity_name(name)


class TestRelationGate:
    @pytest.mark.asyncio
    async def test_temporal_endpoint_skips_the_relation(self) -> None:
        """The write path is the single choke point for relation endpoints."""
        from unittest.mock import AsyncMock, MagicMock

        from graphknows.ingestion.stm._entity_writer import write_relation_edges

        stm = MagicMock()
        stm.write_relation_edge = AsyncMock()
        await write_relation_edges(
            stm,
            "s1",
            [
                {"head": "Melanie", "relation": "plans_to", "tail": "8 May, 2023"},
                {"head": "Melanie", "relation": "attended", "tail": "Pride fest"},
            ],
        )
        assert stm.write_relation_edge.await_count == 1
        assert stm.write_relation_edge.await_args.kwargs["object_name"] == "Pride fest"
