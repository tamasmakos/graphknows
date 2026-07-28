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
            "the",
            "and",
            "or",
            "to",
            "be",
            "These",
            "what",
            "of",
            "a",  # stopwords
            "x",
            "id",
            "io",
            "db",  # single / 2-char lowercase fragments
            "::",
            "...",
            "-",  # punctuation-only
            "http://localhost:5000",
            "www.example.com",  # URLs
            "Flask application: ```python from flask import Flask",  # code
            "config = {a: 1}",
            "arr[]",
            "<div>",  # markup / code
            "December 15, 2023",  # month-first date (previous gap)
            "November 1 - November 15, 2023",  # date range (previous gap)
            "January 16 - February 15, 2024",
            "the quick brown fox jumped over the lazy dog today",  # over-long clause
        ],
    )
    def test_structural_junk_is_dropped(self, name: str) -> None:
        assert not is_graph_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "Flask",
            "SQLAlchemy",
            "Werkzeug",
            "OpenWeatherMap",  # tech entities
            "python",
            "iPhone",  # lowercase / mixed-case real entities
            "AI",
            "US",
            "UK",  # uppercase acronyms survive the 2-char drop
            "May",
            "June",  # month-name collisions with modal/function words
            "San Francisco",
            "New York City",
            "county hospital",  # multi-word names
        ],
    )
    def test_real_entities_survive(self, name: str) -> None:
        assert is_graph_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            # Clause-opener-initial: a pronoun or auxiliary can only start a
            # clause, never head a noun phrase.
            "Been having some tough times lately",
            "It's really made a huge difference",
            "I can teach",
            "since I was a kid",
            "We've",
            "I'm",
            # Utterances: a noun phrase never ends in sentence punctuation.
            "Hey Gina!",
            "Sorry to hear that!",
            "Cool setup!",
            # Backchannel and fixed formulas.
            "Yeah",
            "Wow",
            "Thanks",
            "way to go",
            "me too",
        ],
    )
    def test_clause_fragments_are_not_entities(self, name: str) -> None:
        """Every string here was minted as an ENTITY node on LoCoMo conv-30.

        These reached the GENERATOR, not just the database: the frame layer
        turned role fillers into entities without the junk gate, so the answer
        prompt's fact-sheet carried "Buyer=Hey Gina!", "Speaker=Yeah" and
        "Cognizer=Wow". Grammar decides each case — clause opener, sentence
        punctuation, or closed-class discourse marker — so none can lose a real
        name.
        """
        assert not is_graph_entity_name(name)

    @pytest.mark.xfail(
        reason=(
            "Known defect, measured not fixed: _MAX_ENTITY_WORDS=8 still admits "
            "NOUN-headed and gerund-headed long spans. The clause-opener and "
            "sentence-punctuation rules do not reach them — they are grammatical "
            "noun phrases, just far too long to be entities — and distinguishing "
            "them needs the parse, not the string. The 8-word cap stays: lowering "
            "it to 3 turns these green but deletes legitimate 4-word titles "
            "('Charlotte's Web: A Review'). Unxfail when the minting paths pass "
            "the span's head POS to the gate."
        ),
        strict=True,
    )
    @pytest.mark.parametrize(
        "name",
        [
            "goals written down on paper",
            "keeping me in the loop",
            "one-on-one mentoring and training",
            "new offers and promotions",
        ],
    )
    def test_long_noun_headed_spans_are_still_admitted(self, name: str) -> None:
        assert not is_graph_entity_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "New York City",  # the longest shape a real entity commonly takes
            "county hospital",
            "Dr. Nagy",
            "online store",
            "dance studio",
        ],
    )
    def test_names_at_the_clause_boundary_survive(self, name: str) -> None:
        """Three words is the cut, so the longest ordinary entity names still pass."""
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


def test_first_person_clause_yields_a_relation_from_the_speaker_parameter() -> None:
    """A caller that keeps the speaker OUT of the text must not lose relations.

    Dialogue predication is overwhelmingly first-person ("I work as a nurse"),
    and such a clause only becomes a relation once its subject resolves to a
    name. That resolution used to depend on scraping "ts | Speaker: text" back
    out of the content — so passing clean text plus structured metadata, the
    arrangement that stops speakers being minted as entities, silently cost the
    dominant relation shape. Measured on LoCoMo conv-30: 57 relations with the
    prefix, 15 without. Asserts relations EXIST and the subject is the speaker.
    """
    import spacy

    from graphknows.ingestion.extraction.entities.extractor import _mine_svo_schema
    from graphknows.settings import GraphKnowsSettings

    nlp = spacy.load(GraphKnowsSettings().spacy_model)
    doc = nlp("I work as a nurse at the county hospital.")

    without, _ = _mine_svo_schema(doc)
    assert without == [], "guard: bare first-person text has no resolvable subject"

    with_speaker, _ = _mine_svo_schema(doc, "Gina")
    assert with_speaker, "the speaker parameter must anchor the first-person subject"
    assert any("gina" in str(t).casefold() for t in with_speaker)


def test_scraped_turn_line_speaker_still_wins_for_its_own_sentence() -> None:
    """The parameter is a FALLBACK, not an override.

    A multi-turn chunk has a different speaker per line, which a caller's single
    value cannot express — so a scraped turn-line must keep priority or every
    prefixed corpus would be re-attributed to one person.
    """
    import spacy

    from graphknows.ingestion.extraction.entities.extractor import _mine_svo_schema
    from graphknows.settings import GraphKnowsSettings

    nlp = spacy.load(GraphKnowsSettings().spacy_model)
    doc = nlp("4:04 pm on 20 January, 2023 | Jon: I work as a nurse.")

    triples, _ = _mine_svo_schema(doc, "Gina")

    assert triples
    blob = str(triples).casefold()
    assert "jon" in blob and "gina" not in blob
