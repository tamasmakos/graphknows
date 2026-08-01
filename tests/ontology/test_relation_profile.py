"""The curated relation profile, and property selection by DOMAIN/RANGE.

Two measured failures are what this file locks down.

1. **Whole-chunk cosine cannot select a relation property on dialogue.** Against
   schema.org's 945 object properties, "Still following my passion for dance"
   ranks ``liveBlogUpdate`` (0.64), ``albumRelease`` (0.62) and ``blogPost``
   (0.62) above every property the turn actually asserts. The wrong terms
   OUTSCORE the right ones, so no similarity floor rescues it — end to end it
   produced 0 relations over 60 real turns. Selection is therefore structural:
   the extracted entity TYPES evoke the properties whose declared domain and
   range they satisfy, exactly as a FrameNet trigger evokes its core elements.

2. **A vocabulary can be too small.** Capped at 5 labels the extractor emitted
   ``Gina spouse Jon`` and ``Gina children Jon`` about two friends: with no way
   to decline, a closed vocabulary FORCES a wrong pick. At ~14 offered the same
   turns produced ``colleague``/``owns``/``follows``. So the profile has to be
   big enough to hold near misses, and the per-chunk offer lands in 8-15.

The split this file also pins: schema.org keeps typing ENTITIES (it produced 180
correct INSTANCE_OF edges on a real ingest), while the profile supplies the
RELATION properties. Two indexes, two jobs.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import _RELATION_LABEL_CAP, _STMIngestHandler
from graphknows.ontology.catalog import load_ontology_index
from graphknows.ontology.loader import OntologyTerm, load_ontology_terms
from graphknows.settings import GraphKnowsSettings

PROFILE = Path(GraphKnowsSettings().relation_ontology_source)

# The three terms the research note invents because no surveyed vocabulary has
# an equivalent (schema.org has no like/dislike/plan property on Person, and
# foaf:interest is range Document).
INVENTED = {"likes", "dislikes", "plansTo"}


@pytest.fixture(scope="module")
def terms() -> list[OntologyTerm]:
    return load_ontology_terms(PROFILE)


@pytest.fixture(scope="module")
def index() -> Any:
    idx = load_ontology_index(str(PROFILE))
    assert idx is not None, "the bundled relation profile must load"
    return idx


# ---------------------------------------------------------------------------
# 1. The bundled asset
# ---------------------------------------------------------------------------


class TestProfileAsset:
    def test_load_does_not_import_rdflib(self) -> None:
        """Same contract as the schema.org digest: the base install has no rdflib."""
        code = (
            "import sys, json;"
            "from graphknows.ontology.loader import load_ontology_terms;"
            f"terms = load_ontology_terms({str(PROFILE)!r});"
            "print(json.dumps({'n': len(terms), 'rdflib': 'rdflib' in sys.modules}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout.strip().splitlines()[-1]) == {"n": 34, "rdflib": False}

    def test_profile_size_is_in_the_extractor_s_budget(self, terms: list[OntologyTerm]) -> None:
        """9 classes / 25 properties — GLiNER and GLiREL both train at <=25 labels."""
        assert len([t for t in terms if t.kind == "class"]) == 9
        assert len([t for t in terms if t.kind == "property"]) == 25

    def test_every_term_carries_a_definition(self, terms: list[OntologyTerm]) -> None:
        """The definition IS the label description handed to the extractor."""
        assert all(t.definition for t in terms)

    def test_every_property_is_typed_by_profile_classes(self, terms: list[OntologyTerm]) -> None:
        """Domain/range selection is only possible if every property declares both.

        And both must name a class the profile itself defines, or the property is
        unreachable: an entity type can only be mapped onto a profile class.
        """
        classes = {t.label for t in terms if t.kind == "class"}
        for prop in (t for t in terms if t.kind == "property"):
            assert prop.domain, f"{prop.label} has no domain"
            assert prop.range, f"{prop.label} has no range"
            assert set(prop.domain) <= classes, f"{prop.label} domain outside the profile"
            assert set(prop.range) <= classes, f"{prop.label} range outside the profile"

    def test_reused_terms_keep_their_real_iris(self, terms: list[OntologyTerm]) -> None:
        by_label = {t.label: t.uri for t in terms}
        assert by_label["worksFor"] == "https://schema.org/worksFor"
        assert by_label["Person"] == "https://schema.org/Person"
        # The Relationship vocabulary supplies the two interpersonal terms
        # schema.org lacks entirely.
        assert by_label["friendOf"] == "http://purl.org/vocab/relationship/friendOf"
        assert by_label["lifePartnerOf"] == "http://purl.org/vocab/relationship/lifePartnerOf"

    def test_invented_terms_are_marked_and_namespaced(self, terms: list[OntologyTerm]) -> None:
        """3 of 25 are ours; a reader must not mistake them for schema.org."""
        invented = [t for t in terms if t.label in INVENTED]
        assert len(invented) == len(INVENTED)
        for term in invented:
            assert term.uri.startswith("https://graphknows.io/ontology/personal#")
            assert "Invented" in term.definition
        # Nothing else may claim our namespace.
        assert {t.label for t in terms if "graphknows.io" in t.uri} == INVENTED

    def test_the_class_hierarchy_is_declared(self, terms: list[OntologyTerm]) -> None:
        """Ancestors are what let a Country satisfy a Place range.

        The schema.org digest carries no ``subClassOf`` at all, which silently
        narrows every gate that reads it; the profile is small enough to declare
        its own.
        """
        by_label = {t.label: t for t in terms if t.kind == "class"}
        assert by_label["Country"].parents == ("Place",)
        assert by_label["EducationalOrganization"].parents == ("Organization",)
        assert by_label["Person"].parents == ("Thing",)
        assert by_label["Thing"].parents == ()


# ---------------------------------------------------------------------------
# 2. Selection by domain/range
# ---------------------------------------------------------------------------


def _handler(index: Any, extractor: Any = None, store: Any = None) -> _STMIngestHandler:
    """A handler whose relation vocabulary is the profile (schema.org untouched)."""
    return _STMIngestHandler(
        store=store,
        extractor=extractor,
        embed_fn=lambda _t: [0.0] * 4,
        ontology_index=index,
        relation_index=index,
    )


def _ents(*types: str) -> list[dict[str, Any]]:
    return [{"name": f"e{i}", "type": t} for i, t in enumerate(types)]


class TestDomainRangeSelection:
    def test_a_person_only_chunk_gets_the_interpersonal_and_generic_properties(
        self, index: Any
    ) -> None:
        """One PERSON type is enough to evoke a usable vocabulary — and only that.

        ``worksFor`` and ``attendee`` are deliberately absent: their other
        endpoint (Organization, Event) is nowhere in this chunk, so an extractor
        offered them can only invent one.
        """
        offered = _handler(index)._relation_labels(_ents("PERSON"))
        assert {"knows", "friendOf", "spouse", "likes", "knowsAbout"} <= set(offered)
        assert "worksFor" not in offered
        assert "attendee" not in offered
        # ``owns`` ranges over Product alone — see the range gate below.
        assert "owns" not in offered
        # The measured band: 5 forces a wrong pick, ~14 was precise.
        assert 8 <= len(offered) <= _RELATION_LABEL_CAP

    def test_an_organization_pulls_in_the_employment_properties(self, index: Any) -> None:
        offered = _handler(index)._relation_labels(_ents("PERSON", "ORGANIZATION"))
        assert "worksFor" in offered
        assert "memberOf" in offered

    def test_a_product_pulls_in_the_possession_property(self, index: Any) -> None:
        """``owns`` needs something ownable in the chunk, not just an owner."""
        offered = _handler(index)._relation_labels(_ents("PERSON", "PRODUCT"))
        assert "owns" in offered

    def test_an_event_pulls_in_the_event_properties(self, index: Any) -> None:
        offered = _handler(index)._relation_labels(_ents("PERSON", "EVENT"))
        assert "attendee" in offered  # Event -> Person
        assert "lifeEvent" in offered  # Person -> Event

    def test_a_subclass_type_satisfies_the_parent_s_range(self, index: Any) -> None:
        """``nationality`` ranges over Country; ancestors make Place-ranged terms fire."""
        offered = _handler(index)._relation_labels(_ents("PERSON", "COUNTRY"))
        assert "nationality" in offered
        assert "homeLocation" in offered, "Country is a Place, so a Place range is satisfied"

    def test_the_offer_is_capped_and_prefers_specific_ranges(self, index: Any) -> None:
        """Everything is a Thing, so Thing-ranged properties must not crowd out the rest."""
        offered = _handler(index)._relation_labels(
            _ents("PERSON", "ORGANIZATION", "PLACE", "EVENT", "OCCUPATION")
        )
        assert len(offered) == _RELATION_LABEL_CAP
        for specific in ("worksFor", "hasOccupation", "homeLocation", "attendee"):
            assert specific in offered, f"{specific} lost its slot to a Thing-ranged property"

    def test_the_values_are_definitions_the_extractor_can_use(self, index: Any) -> None:
        offered = _handler(index)._relation_labels(_ents("PERSON"))
        assert "married" in offered["spouse"].lower()

    def test_no_mappable_type_offers_nothing_and_says_so(
        self, index: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silence beats twelve guesses — but a silent silence is unreadable."""
        with caplog.at_level(logging.INFO, logger="graphknows.ingestion.stm.ingest"):
            assert _handler(index)._relation_labels(_ents("QUANTITY", "CARDINAL")) == {}
        assert "no relation property" in caplog.text

    def test_types_present_but_no_property_pairs_them(self, index: Any) -> None:
        """An Organization alone satisfies no property's domain AND range."""
        assert _handler(index)._relation_labels(_ents("ORGANIZATION")) == {}

    def test_no_entities_offers_nothing(self, index: Any) -> None:
        assert _handler(index)._relation_labels([]) == {}


# ---------------------------------------------------------------------------
# 3. Domain/range as a VALIDATOR, against the new profile
# ---------------------------------------------------------------------------


class TestProfileConformanceGates:
    def test_a_person_domain_property_refuses_a_place_subject(self, index: Any) -> None:
        handler = _handler(index)
        assert handler._domain_admits("worksFor", "PERSON") is True
        assert handler._domain_admits("worksFor", "PLACE") is False
        assert handler._domain_blocked("worksFor", "PLACE") is True

    def test_a_person_object_outside_the_range_is_refused(self, index: Any) -> None:
        """``Jon owns students``: the whole reason the range gate exists."""
        handler = _handler(index)
        assert handler._range_admits("owns", "PERSON") is False
        assert handler._range_admits("owns", "PRODUCT") is True

    def test_an_offered_label_maps_to_itself(self, index: Any) -> None:
        """The extractor now emits profile labels verbatim — stage-1 exact match."""
        assert _handler(index)._map_predicate("worksFor", "ORGANIZATION", "PERSON") == (
            "worksFor",
            1.0,
        )

    def test_the_gate_still_annotates_nothing_it_cannot_judge(self, index: Any) -> None:
        handler = _handler(index)
        assert handler._domain_admits("worksFor", "") is True
        assert handler._range_admits("owns", "zzzqqq") is True


# ---------------------------------------------------------------------------
# 4. Two-pass extraction: entities first, then relations with the derived labels
# ---------------------------------------------------------------------------


class _RecordingExtractor:
    """Records every extract() call and returns a fixed entity set."""

    def __init__(self, entity_types: tuple[str, ...] = ("PERSON",)) -> None:
        self.calls: list[dict[str, Any]] = []
        self._types = entity_types

    def extract(
        self,
        text: str,
        extra_entity_labels: Any = None,
        extra_relation_labels: Any = None,
        speaker: str = "",
    ) -> ExtractionResult:
        self.calls.append(
            {"text": text, "entities": extra_entity_labels, "relations": extra_relation_labels}
        )
        return ExtractionResult(
            entities=[{"name": f"E{i}", "type": t} for i, t in enumerate(self._types)]
        )


class _NullStore:
    async def write_chunk(self, chunk: dict[str, Any]) -> None: ...
    async def write_entities_bulk(self, *a: Any) -> None: ...
    async def write_class_matches(self, *a: Any) -> None: ...
    async def write_instance_of(self, *a: Any) -> None: ...
    async def write_temporal(self, *a: Any) -> None: ...
    async def write_relation(self, *a: Any, **kw: Any) -> None: ...


@pytest.mark.asyncio
class TestTwoPassTurnIngest:
    async def test_relations_are_extracted_in_a_second_pass(self, index: Any) -> None:
        """Pass 1 has no relation labels — the types it finds are what select them."""
        extractor = _RecordingExtractor(("PERSON", "ORGANIZATION"))
        handler = _handler(index, extractor=extractor, store=_NullStore())

        await handler.ingest_turn(
            session_id="s", file_id="f", seq=0, role="user", text="I work at Acme.", speaker="Gina"
        )

        assert len(extractor.calls) == 2, "entities first, then relations"
        assert extractor.calls[0]["relations"] is None
        offered = extractor.calls[1]["relations"]
        assert "worksFor" in offered
        assert len(offered) <= _RELATION_LABEL_CAP

    async def test_a_chunk_with_no_selectable_property_is_not_re_extracted(
        self, index: Any
    ) -> None:
        """The second forward pass is only paid for where it can produce something."""
        extractor = _RecordingExtractor(("ORGANIZATION",))
        handler = _handler(index, extractor=extractor, store=_NullStore())

        await handler.ingest_turn(
            session_id="s", file_id="f", seq=0, role="user", text="Acme is hiring.", speaker="Gina"
        )

        assert len(extractor.calls) == 1
