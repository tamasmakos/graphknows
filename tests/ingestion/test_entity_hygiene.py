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
from graphknows.ingestion.extraction.entities.hygiene import (
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
        rel, reason = _build_relation(
            "Caroline: Thank you so much",
            "attended",
            "pride event",
            0.9,
            "relex",
            [],
            speakers=SPEAKERS,
        )
        assert rel is not None, reason
        assert rel["head"] == "Caroline"

    def test_reduction_can_collapse_a_self_relation(self) -> None:
        """(Caroline: hi, told, Caroline) becomes head==tail and is dropped.

        The reason travels with the None so the caller can count it: an unlogged
        swallow makes "found nothing" and "guard ate everything" indistinguishable.
        """
        rel, reason = _build_relation(
            "Caroline: hi", "told", "Caroline", 0.9, "relex", [], speakers=SPEAKERS
        )
        assert rel is None
        assert reason == "self-relation"


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

    @pytest.mark.parametrize(
        "name",
        [
            # Internal sentence punctuation: the span crosses a sentence
            # boundary, so it cannot be one noun phrase.
            "Gina! I",
            "Gina! I won't",
            "Wow! Jon",
            "Really? Gina",
            # Vocative address: a greeting or thanks plus the person addressed.
            "Thanks, Gina",
            "Thank you, Gina",
            "Hey, Jon",
            "Hey Jon",
            "Hi Gina",
            "Congrats, Gina",
            "Sorry, Jon",
        ],
    )
    def test_vocatives_and_split_sentences_are_not_entities(self, name: str) -> None:
        """``Gina! I`` was minted PERSON on LoCoMo conv-30 and carried 11 of the
        graph's 43 relation edges (26%).

        Source text: ``"Thanks, Gina! I won't quit."`` spoken by **Jon** — a
        vocative glued to the following pronoun. The node is neither speaker,
        yet it absorbed every first-person possession Gina uttered. The old
        sentence-punctuation rule only looked at the LAST character, so an
        internal ``!`` walked straight through, as did the bare comma-vocative.
        """
        assert not is_graph_entity_name(name)

    def test_the_conv30_span_is_gated_out_of_the_extraction(self) -> None:
        """The measured spans for ``"Thanks, Gina! I won't quit."`` — Jon speaking.

        ``graph_entities`` is the list-level gate every minting path shares, so
        the whole extraction is asserted by VALUE: the real person survives, the
        vocative-glued fragments do not, and no surviving name carries a ``!``.
        """
        extracted = [
            {"name": "Thanks, Gina! I", "type": "PERSON"},
            {"name": "Gina! I", "type": "PERSON"},
            {"name": "Thanks, Gina", "type": "PERSON"},
            {"name": "Gina", "type": "PERSON"},
        ]
        assert [e["name"] for e in graph_entities(extracted)] == ["Gina"]

    @pytest.mark.parametrize(
        "name",
        [
            "Washington, D.C.",  # comma name whose head segment is no backchannel
            "Charlotte's Web: A Review",
            "Cool Runnings",  # backchannel word, but a title — whole-name rule only
            "Great Expectations",
            "Gina",
            "Jon",
            "dance studio",
        ],
    )
    def test_vocative_rules_do_not_touch_real_names(self, name: str) -> None:
        assert is_graph_entity_name(name)

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

        from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
        from graphknows.ingestion.stm.ingest import _STMIngestHandler

        store = MagicMock()
        store.write_relation = AsyncMock()
        handler = _STMIngestHandler(store, extractor=None, embed_fn=MagicMock())
        await handler._write_relations(
            ExtractionResult(
                relations=[
                    {"head": "Melanie", "relation": "plans_to", "tail": "8 May, 2023"},
                    {"head": "Melanie", "relation": "attended", "tail": "Pride fest"},
                ]
            ),
            chunk_id="c1",
            text="Melanie attended Pride fest on 8 May, 2023.",
        )
        assert store.write_relation.await_count == 1
        assert store.write_relation.await_args.args[2] == "Pride fest"


def test_speaker_parameter_reaches_the_schema_when_the_text_omits_it() -> None:
    """A caller that keeps the speaker OUT of the text must still name the speaker.

    ``_DynamicSchema.speakers`` is what ``_reduce_speaker_span`` keys on to fold
    a span that ran across a ``"Speaker:"`` colon back onto the person — the
    mechanism that stopped 27 malformed names carrying 42.5% of REL edges. A
    caller passing clean text plus structured metadata (the arrangement that
    stops speakers being minted as entities) has no turn-line to scrape, so the
    parameter is the only source.
    """
    from graphknows.ingestion.extraction.entities.extractor import _SpacySchemaMiner

    miner = _SpacySchemaMiner()

    assert miner.build("I work as a nurse at the county hospital.").speakers == frozenset()
    assert miner.build("I work as a nurse.", "Gina").speakers == frozenset({"gina"})


def test_scraped_turn_line_speakers_join_the_caller_supplied_one() -> None:
    """A multi-turn chunk has a speaker per line, which one caller value cannot express."""
    from graphknows.ingestion.extraction.entities.extractor import _SpacySchemaMiner

    schema = _SpacySchemaMiner().build(
        "4:04 pm on 20 January, 2023 | Jon: I work as a nurse.\n"
        "4:06 pm on 20 January, 2023 | Melanie: Nice.",
        "Gina",
    )

    assert schema.speakers == frozenset({"jon", "melanie", "gina"})


class TestVerbSurfacesAreNotEntities:
    """Predicate spans returned as entities are dropped — using CONTEXT, not a lexicon.

    Measured on conv-30 (namespace ``lgdemo``, 941 entities): of the 912 whose
    mention could be located in a chunk, 164 (18.0%) are verb-headed where they
    actually occur — ``starting``, ``take``, ``Lost``, ``speaks``, ``explore``,
    ``Winning`` — and the extractor types most of them PERSON (79) or EVENT (52).

    The reason this reads the doc rather than the name: tagging those same names
    in ISOLATION calls a further 67 of them (7.3%) verbs when the parser reads
    them as nouns in their own sentence — ``shot`` in "take a shot at starting",
    ``work`` in "after work", ``dances`` in "I love all dances", ``moves`` in
    "see my moves". A lexicon or a string rule deletes exactly those. An earlier
    pass of this audit reported 26.1% from the isolated tagging; the in-context
    figure is 18.0%.
    """

    @staticmethod
    def _doc(text: str):
        import spacy

        from graphknows.settings import GraphKnowsSettings

        return spacy.load(GraphKnowsSettings().spacy_model, disable=["ner"])(text)

    def test_verb_only_surface_is_mined(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _mine_verb_surfaces

        surfaces = _mine_verb_surfaces(self._doc("I explore the city and she speaks Italian."))
        assert "explore" in surfaces
        assert "speaks" in surfaces
        assert "city" not in surfaces

    def test_surface_used_as_a_noun_is_not_mined(self) -> None:
        """`work` is a verb AND a noun here — mining it would delete a real entity."""
        from graphknows.ingestion.extraction.entities.extractor import _mine_verb_surfaces

        surfaces = _mine_verb_surfaces(self._doc("I work at the store after work."))
        assert "work" not in surfaces

    def test_noun_reading_survives(self) -> None:
        """The 7.3% an isolated-tagging rule would wrongly drop."""
        from graphknows.ingestion.extraction.entities.extractor import _mine_verb_surfaces

        surfaces = _mine_verb_surfaces(
            self._doc("Lost my job as a banker, so I'm gonna take a shot at it.")
        )
        assert "shot" not in surfaces

    def test_dedupe_drops_a_verb_surface(self) -> None:
        raw = [
            {"text": "speaks", "label": "person", "score": 0.9},
            {"text": "Maria", "label": "person", "score": 0.9},
        ]
        out = _dedupe_entities(raw, {"person": "PERSON"}, frozenset(), frozenset({"speaks"}))
        assert [e["name"] for e in out] == ["Maria"]

    def test_dedupe_without_verb_surfaces_is_unchanged(self) -> None:
        raw = [{"text": "speaks", "label": "person", "score": 0.9}]
        assert len(_dedupe_entities(raw, {"person": "PERSON"}, frozenset())) == 1


class TestDeterminerStripping:
    """A leading article is not part of the entity, and it forks the merge key.

    Measured on conv-30: 31 names carry one (``a banker``, ``the ideal spot``,
    ``The process``), each becoming a second ENTITY node for a referent that
    already has one.
    """

    def test_article_is_stripped(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _strip_determiner

        assert _strip_determiner("a banker") == "banker"
        assert _strip_determiner("The process") == "process"
        assert _strip_determiner("an owl") == "owl"

    def test_non_article_prefix_is_untouched(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _strip_determiner

        assert _strip_determiner("Andrew Marr") == "Andrew Marr"
        assert _strip_determiner("Theatre Royal") == "Theatre Royal"
        assert _strip_determiner("Ann") == "Ann"

    def test_dedupe_merges_the_determiner_variant(self) -> None:
        """`a banker` and `banker` are one referent, so one node."""
        raw = [
            {"text": "a banker", "label": "occupation", "score": 0.9},
            {"text": "banker", "label": "occupation", "score": 0.8},
        ]
        out = _dedupe_entities(raw, {"occupation": "OCCUPATION"}, frozenset())
        assert [e["name"] for e in out] == ["banker"]


class TestSupersenseTypesReachTheTaxonomy:
    """A relation endpoint must not carry WordNet's vocabulary into ENTITY.type.

    ``_arg_type`` falls back to a WordNet supersense when NER does not label a
    relation argument, and ``ENTITY.type`` is first-write-wins, so an entity
    first seen as a relation endpoint kept that supersense permanently. Measured
    on conv-30 (namespace lg30v3): EVERY entity whose type sat outside the
    taxonomy was a relation endpoint — ``clothes store`` noun.artifact,
    ``customer base`` noun.artifact, ``stress buster`` noun.person, ``target
    audience`` noun.group.

    Two vocabularies in one column make ENTITY.type unqueryable, and the
    supersense form also reaches the ontology class matcher as an unknown label,
    where it can only ever score by embedding (``noun.person`` -> Person 0.840
    rather than the 1.000 a label hit gives).
    """

    def test_supersense_is_translated(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _taxonomy_type

        assert _taxonomy_type("noun.person") == "PERSON"
        assert _taxonomy_type("noun.group") == "ORGANIZATION"
        assert _taxonomy_type("noun.artifact") == "ARTIFACT"

    def test_unmapped_supersense_goes_untyped_not_wrong(self) -> None:
        """ "" is the safe answer: coalesce lets a later NER write fill it in."""
        from graphknows.ingestion.extraction.entities.extractor import _taxonomy_type

        assert _taxonomy_type("noun.tops") == ""
        assert _taxonomy_type("") == ""

    def test_arg_type_never_returns_wordnet_vocabulary(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _arg_type

        # Not in the entity list -> falls through to the supersense path.
        assert not _arg_type("students", []).startswith("noun.")
        assert _arg_type("students", []) == "PERSON"

    def test_ner_label_still_wins_over_the_supersense(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _arg_type

        entities = [{"name": "Maria", "type": "PERSON"}]
        assert _arg_type("Maria", entities) == "PERSON"


class TestRelationEndpointProvenance:
    """write_relation gives its endpoints chunk provenance — but not the speaker.

    An entity that only ever appears as a relation argument had no MENTIONS edge
    at all: measured on conv-30, 4 of 49 endpoints ('clothes store', 'running',
    'stress buster', 'target audience') were reachable from no CHUNK, so nothing
    could explain where they came from.

    The chunk's SPEAKER is deliberately excluded. A relation head is usually the
    speaker resolved from metadata ("my collection" -> Gina), and CHUNK.speaker
    already carries that — typed, and stronger than MENTIONS. An unfiltered
    version of this fix added 62 MENTIONS edges to Jon and Gina on conv-30, the
    two most saturated entities in the graph and exactly the ones retrieval has
    to de-weight (see _discriminative_entities).
    """

    @staticmethod
    def _norms_written(commands):
        for sql, kwargs in commands:
            if "MENTIONS" in sql:
                return kwargs.get("norms")
        return None

    async def test_endpoints_get_mentions_but_speaker_does_not(self) -> None:
        from graphknows.storage.arcadedb.graph_store import GraphStore

        calls: list[tuple[str, dict]] = []

        class _Store(GraphStore):
            def __init__(self) -> None:  # no connection
                pass

            async def command(self, sql, **kwargs):  # type: ignore[override]
                calls.append((sql, kwargs))
                return []

        await _Store().write_relation("Gina", "HAS", "collection", chunk_id="c1", speaker="Gina")
        assert self._norms_written(calls) == ["collection"], (
            "the speaker must not gain a MENTIONS edge from its own chunk"
        )

    async def test_both_endpoints_when_neither_is_the_speaker(self) -> None:
        from graphknows.storage.arcadedb.graph_store import GraphStore

        calls: list[tuple[str, dict]] = []

        class _Store(GraphStore):
            def __init__(self) -> None:
                pass

            async def command(self, sql, **kwargs):  # type: ignore[override]
                calls.append((sql, kwargs))
                return []

        await _Store().write_relation("Maria", "HAS", "lease", chunk_id="c1", speaker="Gina")
        assert self._norms_written(calls) == ["maria", "lease"]

    async def test_no_chunk_id_writes_no_mentions(self) -> None:
        from graphknows.storage.arcadedb.graph_store import GraphStore

        calls: list[tuple[str, dict]] = []

        class _Store(GraphStore):
            def __init__(self) -> None:
                pass

            async def command(self, sql, **kwargs):  # type: ignore[override]
                calls.append((sql, kwargs))
                return []

        await _Store().write_relation("Maria", "HAS", "lease", speaker="Gina")
        assert self._norms_written(calls) is None


class TestUtteranceBoundarySpans:
    """A span running across a sentence or a speaker line is two fragments, not a name.

    Introduced by multi-turn extraction (``GRAPHKNOWS_EXTRACT_WINDOW``): with a
    window of consecutive turns the model can open a span in one turn and close
    it in the next. Measured on conv-30 — 2/733 such names (0%) when turns were
    extracted alone, 10/282 (4%) with a 5-turn window — and they arrive through
    RELATION ENDPOINTS as well as the entity path, which is why the rule lives
    in ``is_graph_entity_name`` where both routes pass through it.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "business. Gina: Sorry about your job Jon",  # the measured case
            "Jon. Hang in there and Jon",
            "Gina: Sorry about your job",  # speaker colon alone
            "I lost my job. Starting a business",  # sentence break alone
            "dance studio\nGina",  # the window's own newline join
        ],
    )
    def test_a_span_crossing_a_boundary_is_refused(self, name: str) -> None:
        assert is_graph_entity_name(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "Dr. Nagy",  # title abbreviation
            "U.S. Army",  # initialism
            "J. R. Ewing",  # initials
            "St. Louis",
            "Mochi",
            "dance studio",
            "Door Dash",
        ],
    )
    def test_a_real_name_with_periods_survives(self, name: str) -> None:
        """The rule must key on SENTENCE-ending periods, not on any period at all."""
        assert is_graph_entity_name(name) is True
