"""Domain/range conformance of extracted relations, and its per-chunk report.

``test_injection_quality`` covers the RANGE half of the gate (object side). This
file covers the half that was missing — the DOMAIN (subject side) — plus the
counters that make either half auditable.

Why the two halves are not enforced symmetrically:

The range gate concludes "the object IS a person", so it needs the specific
``-> Person`` mapping to be right, and a wrong one (``PRODUCT -> Person`` at
0.580) would delete a correct edge. Hence its confidence floor.

The domain gate concludes "the subject is NOT a person", which every non-person
class supports equally — it does not matter whether ``GPE`` landed on ``Event``
(0.494) or ``Place``; either way it is not a person. Measured against
personal.ttl with the baked embedder:

    PERSON        -> Person        1.000
    noun.person   -> Person        0.840
    PRODUCT       -> Person        0.580   (wrong, but admits: safe direction)
    LOCATION      -> Place         0.653
    noun.location -> Place         0.843
    ORGANIZATION  -> Organization  1.000
    OCCUPATION    -> Activity      0.613
    GPE           -> Event         0.494

No type that denotes a person maps to a non-person class, so the domain gate
needs no floor beyond ``_ONT_CLASS_MIN_SIM`` (0.45), which ``_map_to_class``
already applies before it returns a class at all.

Like the range gate this ANNOTATES rather than filters: a blocked relation is
still written under its extracted predicate, with ``ontology_property`` dropped
and ``ontology_blocked`` set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.storage.arcadedb.graph_store import RelationWrite
from graphknows.symbolic.ontology.catalog import load_ontology_index

PERSONAL_TTL = Path(__file__).resolve().parents[1] / "fixtures" / "ontology" / "personal.ttl"


@pytest.fixture(scope="module")
def index():
    idx = load_ontology_index(str(PERSONAL_TTL))
    assert idx is not None, "personal.ttl must load — the rest of this file is meaningless"
    return idx


class _FakeStore:
    """Records every write instead of performing it."""

    def __init__(self) -> None:
        self.relations: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.entities: list[Any] = []
        self.groundings: list[Any] = []
        self.class_matches: list[Any] = []
        self.temporal: list[Any] = []

    async def write_chunk(self, chunk: dict[str, Any]) -> None:
        self.chunks.append(chunk)

    async def write_entities_bulk(self, session_id, chunk_id, entities) -> None:
        self.entities.extend(entities)

    async def write_class_matches(self, chunk_id, matches) -> None:
        self.class_matches.append((chunk_id, matches))

    async def write_groundings(self, mappings) -> None:
        self.groundings.extend(mappings)

    async def write_temporal(self, chunk_id, iso_dates) -> None:
        self.temporal.append((chunk_id, iso_dates))

    async def write_relation(self, write: RelationWrite) -> None:
        self.relations.append(
            {
                "head": write.src_name,
                "relation": write.predicate,
                "tail": write.dst_name,
                "ontology_property": write.ontology_property,
                "ontology_blocked": write.ontology_blocked,
            }
        )

    def annotated(self) -> dict[tuple[str, str], str]:
        """``(head, tail) -> ontology_property`` for every written relation."""
        return {(r["head"], r["tail"]): r.get("ontology_property", "") for r in self.relations}

    def blocked(self) -> set[tuple[str, str]]:
        return {(r["head"], r["tail"]) for r in self.relations if r.get("ontology_blocked")}


def _handler(index, store=None):
    # Same index for both jobs: these assertions are about personal.ttl's own
    # domain/range declarations, so the relation vocabulary must be that file
    # rather than the bundled profile ``relation_index=None`` would load.
    return _STMIngestHandler(
        store=store, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )


# ---------------------------------------------------------------------------
# 1. The domain half of the gate — the spec's "violating the declared domain"
# ---------------------------------------------------------------------------


def test_declared_domain_blocks_a_subject_the_property_cannot_have(index):
    """``London works_as Analytical Engine`` must not be annotated ``works_as``.

    ``:works_as rdfs:domain :Person``, and "London" types LOCATION, which maps
    to ``Place`` (0.653). A place cannot be the subject of a Person-domain
    property, so the annotation is refused — the exact mirror of the range gate.
    """
    handler = _handler(index)

    assert handler._map_predicate("works for", "ORGANIZATION", "LOCATION") == ("", 0.0)
    assert handler._domain_blocked("works for", "LOCATION") is True

    # The conforming subject keeps the annotation (spec: "a conforming relation
    # is kept").
    prop, score = handler._map_predicate("works for", "ORGANIZATION", "PERSON")
    assert prop == "works_as", f"a Person subject must satisfy the domain, got {prop!r}"
    assert score > 0.0
    assert handler._domain_blocked("works for", "PERSON") is False


def test_domain_gate_admits_what_it_cannot_judge(index):
    """Enrichment, never a filter: silence on every uncertain subject."""
    handler = _handler(index)

    # Unknown subject type -> nothing to judge.
    assert handler._map_predicate("works for", "ORGANIZATION", "")[0] == "works_as"
    # A type that maps to no ontology class at all -> nothing to judge.
    assert handler._domain_blocked("works for", "zzzqqq") is False
    # A property whose domain is not Person is left alone: only the Person-domain
    # case has a measured defect behind it.
    assert handler._domain_blocked("happened at", "PERSON") is False
    # An unmappable predicate is silence, not a refusal (mirrors _range_blocked).
    assert handler._domain_blocked("photosynthesizes", "LOCATION") is False


def test_no_ontology_configured_never_blocks_on_domain():
    handler = _handler(None)
    assert handler._domain_blocked("works for", "LOCATION") is False


def test_a_supersense_subject_is_judged_like_any_other(index):
    """Relation endpoints typed by WordNet supersense reach the gate too.

    ``noun.location`` maps to ``Place`` at 0.843; ``noun.person`` to ``Person``
    at 0.840. The first must block a Person-domain property, the second must not.
    """
    handler = _handler(index)
    assert handler._domain_blocked("lives in", "noun.location") is True
    assert handler._domain_blocked("lives in", "noun.person") is False


# ---------------------------------------------------------------------------
# 2. The per-chunk conformance / hallucination report
# ---------------------------------------------------------------------------


def _rel(head, relation, tail, head_type="PERSON", tail_type="PERSON"):
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "head_type": head_type,
        "tail_type": tail_type,
        "score": 0.7,
    }


@pytest.mark.asyncio
async def test_conformance_and_hallucination_rates_are_reported_per_chunk(index):
    """The spec's worked example: 10 relations, 2 pruned, 1 unsupported.

    Then conformance is 0.8 and hallucination is 0.1.
    """
    store = _FakeStore()
    handler = _handler(index, store)

    people = ["Ada", "Charles", "Mary", "Grace", "Alan", "Edith", "Joan"]
    extraction = ExtractionResult(
        entities=[
            *[{"name": n, "type": "PERSON"} for n in [*people, "Maria"]],
            {"name": "London", "type": "LOCATION"},
            {"name": "Analytical Engine", "type": "ORGANIZATION"},
        ],
        relations=[
            # 7 conforming: Person friend_of Person.
            *[_rel("Ada", "is friends with", p) for p in people],
            # 1 domain violation: works_as has domain Person, London is a Place.
            _rel(
                "London",
                "works for",
                "Analytical Engine",
                head_type="LOCATION",
                tail_type="ORGANIZATION",
            ),
            # 1 range violation: owns has range Possession, Maria is a person.
            _rel("Ada", "HAS", "Maria"),
            # 1 unsupported: "Bletchley" is in no extracted span.
            _rel("Ada", "is friends with", "Bletchley"),
        ],
    )

    report = await handler._write_relations(extraction, "chunk-1", "some text", "Ada")

    assert report["relations"] == 10
    assert report["blocked_domain"] == 1
    assert report["blocked_range"] == 1
    assert report["unsupported"] == 1
    assert report["conformance"] == pytest.approx(0.8)
    assert report["hallucination"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_the_resolved_speaker_is_a_supported_endpoint(index):
    """A speaker-resolved head is not a hallucination — it is the feature working.

    SVO resolves a first-person subject to the SPEAKER, so "My collection has
    twenty pieces" spoken by Gina yields head "Gina" — a name that by design
    appears in no extracted span of that chunk. Counting it unsupported makes
    the hallucination rate read ~1.0 on every conversational chunk regardless
    of quality, which is a metric that measures nothing.

    Measured before this fix, on a live ingest of three fixture turns with
    personal.ttl loaded: every chunk logged hallucination=1.000.
    """
    store = _FakeStore()
    handler = _handler(index, store)

    extraction = ExtractionResult(
        entities=[{"name": "collection", "type": "PRODUCT"}],
        relations=[_rel("Gina", "HAS", "collection")],
    )
    report = await handler._write_relations(
        extraction, "chunk-spk", "My collection has pieces.", "Gina"
    )

    assert report["relations"] == 1
    assert report["unsupported"] == 0, "the speaker is a legitimate endpoint, not a hallucination"
    assert report["hallucination"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_an_endpoint_in_neither_the_spans_nor_the_speaker_is_still_unsupported(index):
    """The fix must not blind the metric: a genuinely invented endpoint still counts."""
    store = _FakeStore()
    handler = _handler(index, store)

    extraction = ExtractionResult(
        entities=[{"name": "collection", "type": "PRODUCT"}],
        relations=[_rel("Gina", "HAS", "Bletchley")],
    )
    report = await handler._write_relations(extraction, "chunk-spk2", "My collection.", "Gina")

    assert report["unsupported"] == 1
    assert report["hallucination"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_an_empty_chunk_report_is_perfectly_conforming_not_undefined(index):
    """No relations means nothing violated — 1.0/0.0, never a ZeroDivisionError."""
    report = await _handler(index, _FakeStore())._write_relations(
        ExtractionResult(), "chunk-0", "", ""
    )
    assert report == {
        "relations": 0,
        "pruned_endpoint": 0,
        "blocked_domain": 0,
        "blocked_range": 0,
        "unsupported": 0,
        "conformance": 1.0,
        "hallucination": 0.0,
    }


# ---------------------------------------------------------------------------
# 3. End to end through the real ingest path
# ---------------------------------------------------------------------------


class _StubExtractor:
    """Returns a fixed extraction so the ASSERTIONS are about the ontology layer."""

    def __init__(self, result: ExtractionResult) -> None:
        self._result = result

    def extract(self, text, entity_labels=None, relation_labels=None, speaker="") -> Any:
        return self._result


@pytest.mark.asyncio
async def test_ingesting_a_turn_annotates_conforming_relations_and_refuses_the_rest(index):
    """personal.ttl + three fixture relations, asserted by VALUE.

    ``Ada works_as Analytical Engine``  -> annotated ``works_as``
    ``London works for Analytical Engine`` -> refused (domain: Place, not Person)
    ``Ada HAS Maria``                   -> refused (range: Person, not Possession)
    """
    extraction = ExtractionResult(
        entities=[
            {"name": "Ada", "type": "PERSON"},
            {"name": "London", "type": "LOCATION"},
            {"name": "Maria", "type": "PERSON"},
            {"name": "Analytical Engine", "type": "ORGANIZATION"},
        ],
        relations=[
            _rel("Ada", "works for", "Analytical Engine", tail_type="ORGANIZATION"),
            _rel(
                "London",
                "works for",
                "Analytical Engine",
                head_type="LOCATION",
                tail_type="ORGANIZATION",
            ),
            _rel("Ada", "HAS", "Maria"),
        ],
    )
    store = _FakeStore()
    handler = _STMIngestHandler(
        store=store,
        extractor=_StubExtractor(extraction),
        ontology_index=index,
        relation_index=index,
    )

    await handler.ingest_turn(
        session_id="s1",
        file_id="f1",
        seq=0,
        role="user",
        text="Ada works for the Analytical Engine and her sister Maria lives in London.",
        speaker="Ada",
    )

    assert store.annotated() == {
        ("Ada", "Analytical Engine"): "works_as",
        ("London", "Analytical Engine"): "",
        ("Ada", "Maria"): "",
    }
    assert store.blocked() == {("London", "Analytical Engine"), ("Ada", "Maria")}
