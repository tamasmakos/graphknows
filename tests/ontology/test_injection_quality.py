"""Ontology injection quality — does definition-matching surface the RIGHT terms?

``test_ontology_terms`` covers the mechanics (facets stay separate, top-k is
bounded, a term self-matches). This file covers the semantics: given ordinary
conversational prose, does the ontology term a human would pick actually rank
above one they would not?

Assertions are deliberately **pairwise and wide-margin**. Absolute cosines are
uncalibrated and model-dependent (see the catalog module docstring: a wrong
class can score 0.72), and rank positions shift between embedding models, so
"works_as beats dislikes for a sentence about a job" is a claim any competent
embedder should satisfy whereas "works_as ranks first" is not.

Every case below holds under both ``BAAI/bge-small-en-v1.5`` and
``zeroentropy/zembed-1-embedding``.

Class matching used to be the weak half — under zembed-1 ``Possession`` ranked
11/11 for "bought that vintage motorcycle", behind ``Emotion``. That was not the
ontology's fault but an embedder bug: the model declares a ``suffix`` that
sentence-transformers silently ignores, and it pools the last token. See
``tests/storage/test_embedder_suffix.py``. The class cases here are kept partly
as a high-level guard on that fix.

Class matches still deserve the closer watch: they drive entity-label hints *and*
the persisted ``MATCHES_CLASS`` tags, so a wrong class is written to the graph,
whereas a wrong property candidate merely goes unused when the extractor finds no
span for it. Margins on the class side also remain thinner than on the property
side even when correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphknows.ontology.catalog import load_ontology_index
from graphknows.storage.embedder import embed_one

PERSONAL_TTL = (
    Path(__file__).resolve().parents[2]
    / "graphknows"
    / "ontology"
    / "assets"
    / "personal"
    / "personal.ttl"
)


@pytest.fixture(scope="module")
def index():
    idx = load_ontology_index(str(PERSONAL_TTL))
    assert idx is not None, "personal.ttl must load — the rest of this file is meaningless"
    return idx


def _score(matches, label: str) -> float:
    """Cosine for ``label`` within a ranked match list."""
    for term, score in matches:
        if term.label == label:
            return score
    raise AssertionError(f"{label!r} absent from matches: {[t.label for t, _ in matches]}")


# (sentence, expected property, property a human would NOT pick)
PROPERTY_CASES = [
    (
        "Caroline has been working as a nurse at the children's hospital for six years.",
        "works_as",
        "dislikes",
    ),
    (
        "My sister moved to Berlin last spring and rents a flat near the park.",
        "lives_in",
        "works_as",
    ),
    (
        "Marcus and I have been close friends since we were at university together.",
        "friend_of",
        "works_as",
    ),
    (
        "We all went to the jazz concert on Friday night and it was fantastic.",
        "attended",
        "works_as",
    ),
    (
        "He finally bought that vintage motorcycle he had been saving up for.",
        "owns",
        "friend_of",
    ),
]


@pytest.mark.parametrize(("text", "expected", "irrelevant"), PROPERTY_CASES)
def test_relevant_property_outranks_irrelevant_one(index, text, expected, irrelevant):
    """The property a human would pick must beat one they plainly would not."""
    matches = index.match_properties(embed_one(text), top_k=99, min_sim=-1.0)
    assert _score(matches, expected) > _score(matches, irrelevant), (
        f"{expected!r} did not outrank {irrelevant!r} for: {text}"
    )


# (sentence, expected class, class a human would NOT pick)
#
# These three were the observable symptom of the last-token-pooling bug: under
# zembed-1, "Possession" ranked 11/11 and "Place" 7/11 until the model's declared
# suffix was applied (see tests/storage/test_embedder_suffix.py). They now hold
# under both bge-small and zembed-1, so they double as a high-level guard on that
# fix — if the suffix stops being applied, these go red.
CLASS_CASES = [
    (
        "Marcus and I have been close friends since we were at university together.",
        "Relationship",
        "TimePoint",
    ),
    (
        "My sister moved to Berlin last spring and rents a flat near the park.",
        "Place",
        "Emotion",
    ),
    (
        "He finally bought that vintage motorcycle he had been saving up for.",
        "Possession",
        "Emotion",
    ),
]


@pytest.mark.parametrize(("text", "expected", "irrelevant"), CLASS_CASES)
def test_relevant_class_outranks_irrelevant_one(index, text, expected, irrelevant):
    """Same contract on the class side — these steer entity labels, not predicates."""
    matches = index.match_classes(embed_one(text), top_k=99, min_sim=-1.0)
    assert _score(matches, expected) > _score(matches, irrelevant), (
        f"{expected!r} did not outrank {irrelevant!r} for: {text}"
    )


def test_small_ontology_is_injected_whole_at_the_production_top_k(index):
    """Ranking is load-bearing now; for a small ontology the FLOOR is what bites.

    This test used to assert ``DEFAULT_MIN_SIM == 0.0`` — injection gated by
    top-k alone — which was survivable only because personal.ttl has 11 classes,
    so top-12 was the whole ontology. The default ontology is schema.org's 979
    classes, so there is a floor (0.50, measured over real LoCoMo turns), and
    against a 13-property ontology it is the floor, not the budget, that decides
    what reaches the extractor.
    """
    from graphknows.ontology.catalog import DEFAULT_MIN_SIM, DEFAULT_TOP_K

    assert DEFAULT_MIN_SIM == 0.50

    emb = embed_one("Some ordinary sentence about nothing in particular.")
    unfloored = index.match_classes(emb, top_k=DEFAULT_TOP_K, min_sim=-1.0)
    assert len(unfloored) == len(index.classes), "every class fits the budget"

    floored = index.match_classes(emb, top_k=DEFAULT_TOP_K)
    assert [t.label for t, _ in floored] == [t.label for t, s in unfloored if s >= DEFAULT_MIN_SIM]


# ---------------------------------------------------------------------------
# Known gaps — recorded as strict xfail so they announce themselves when fixed
# ---------------------------------------------------------------------------

# ``lives_in`` behaves as a magnet: it reaches the top 3 for 5 of 7 probe
# sentences and takes first place on two where it is plainly wrong. Harmless
# today because a 13-property ontology is injected whole (see the top-k test
# above), but it would suppress the right term under a budget-constrained
# ontology. Same shape as the entity-saturation problem on the retrieval side.
LIVES_IN_MAGNET = [
    (
        "She really loves hiking in the mountains but cannot stand crowded beaches.",
        "likes",
    ),
    (
        "I am planning to visit my parents in Madrid next December.",
        "plans_to",
    ),
]


# Non-strict deliberately: the gap is embedder-dependent. Under bge-small both
# cases fail; under zembed-1 the "likes" case now passes and "plans_to" still
# fails. A strict xfail would therefore turn green on one supported model into a
# suite failure, which is a property of the test rig rather than of the code.
@pytest.mark.xfail(reason="'lives_in' outranks the correct property (magnet term)")
@pytest.mark.parametrize(("text", "expected"), LIVES_IN_MAGNET)
def test_known_gap_lives_in_crowds_out_the_correct_property(index, text, expected):
    matches = index.match_properties(embed_one(text), top_k=99, min_sim=-1.0)
    assert _score(matches, expected) > _score(matches, "lives_in")


def test_predicate_mapper_maps_surface_forms_to_ontology_properties(index):
    """The extract-then-map wiring: free-form predicates → the ontology vocabulary.

    Mirrors how ``STMService`` builds the mapper — from the loaded index's
    ``property_labels`` — so this covers the real production seam rather than a
    hand-written label list.
    """
    from graphknows.ingestion.extraction.relations.filter import OntologyPredicateMapper

    mapper = OntologyPredicateMapper(index.property_labels)

    assert mapper.map_predicate("friend_of")[0] == "friend_of"  # exact
    assert mapper.map_predicate("is friends with")[0] == "friend_of"  # surface alias
    assert mapper.map_predicate("works for")[0] == "works_as"
    assert mapper.map_predicate("belongs to")[0] == "member_of"

    unmapped, score = mapper.map_predicate("photosynthesizes")
    assert unmapped is None, f"expected unmappable, got {unmapped!r} at {score:.3f}"


def test_declared_range_blocks_a_predicate_mapping_the_object_cannot_satisfy(index):
    """``Gina HAS Maria`` must not be recorded as ``Gina owns Maria``.

    ``_SURFACE_ALIASES`` maps the bare predicate "has" onto ``owns``, which is
    right for a possession and wrong for a person. The ontology already carries
    the constraint that settles it -- ``:owns rdfs:range :Possession`` -- but the
    mapper is a pure string matcher and never saw the object.

    Measured on the 5-turn ingest smoke (namespace ``smoke``) before the gate:
    all three extracted relations mapped to ``owns`` at ``map_score=0.95``,
    including ``Jon HAS mother`` and ``Gina HAS Maria`` -- confident, persisted,
    false assertions about people. ``Jon HAS dance studio`` is the one the
    alias gets right, so the gate must keep it.

    The gate ANNOTATES, never filters: a blocked mapping drops the
    ``ontology_property`` only. The relation is still written under its
    extracted predicate.
    """
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )

    # A person object cannot satisfy `owns` (range: Possession) -> no mapping.
    assert handler._map_predicate("HAS", "PERSON") == ("", 0.0)
    # The same predicate over a possession still maps.
    prop, score = handler._map_predicate("HAS", "POSSESSION")
    assert prop == "owns", f"expected owns for a possession object, got {prop!r}"
    assert score > 0.0

    # Unknown object type -> admit. Enrichment must never become a filter.
    assert handler._map_predicate("HAS", "")[0] == "owns"

    # A property with no declared range is unaffected by the gate.
    assert handler._map_predicate("is friends with", "PERSON")[0] == "friend_of"


@pytest.mark.parametrize("object_type", ["PLACE", "PRODUCT", "ORGANIZATION", "OCCUPATION"])
def test_range_gate_keeps_mappings_the_ontology_is_merely_too_narrow_for(index, object_type):
    """A non-person object keeps ``owns`` even when it misses the declared range.

    The first cut of this gate enforced every ``rdfs:range`` and was measured far
    too strict. On conv-30 (369 turns, namespace ``lg30o``, ontology on) there
    are 27 ``owns`` edges; 26 are correct — ``Gina -> collection``, ``-> store``,
    ``-> fashion pieces``, ``-> business``, ``-> online store`` — and their types
    are PLACE / PRODUCT / ORGANIZATION / OCCUPATION, none of which satisfy
    ``:owns rdfs:range :Possession``. Full enforcement dropped all 26 to catch
    the single genuine error (``Jon owns students``).

    So the ontology is narrower than the data, and a range miss is NOT by itself
    evidence of a wrong assertion. Only a CONFIDENT person-object is.

    PRODUCT is the case that pins the confidence floor rather than the class
    rule: the type->class embedding maps PRODUCT onto Person at 0.580, which
    clears ``_ONT_CLASS_MIN_SIM`` (0.45). Gating on the class alone therefore
    still deleted ``Gina owns fashion pieces`` and ``Gina owns online store``.
    Requiring a confident match leaves them.

    Measured over all 37 annotated edges in lg30v2: 35 kept, 2 blocked.
    """
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )
    assert handler._map_predicate("HAS", object_type)[0] == "owns"


def test_range_gate_fires_on_a_wordnet_supersense_person(index):
    """A relation endpoint typed by supersense is still a person.

    ``_arg_type`` falls back to a WordNet supersense when NER does not label the
    argument, and a supersense never label-matches an ontology class — so
    ``noun.person`` reaches the gate at an EMBEDDING score (0.840), not 1.0.
    A confidence floor set from the label-hit score alone (0.9) sat above it and
    let through ``Jon owns students`` — the one edge in all of conv-30 that this
    gate exists to stop. It survived a full re-ingest before this was found.

    Censused over all 20 entity/endpoint types in conv-30, exactly three map to
    Person: PERSON 1.000 and noun.person 0.840, both correct, against PRODUCT
    0.580, wrong. The floor sits at the midpoint of that gap.
    """
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )
    assert handler._map_predicate("HAS", "noun.person") == ("", 0.0)
    # ...while the WRONG Person mapping stays below the floor and is left alone.
    assert handler._map_predicate("HAS", "PRODUCT")[0] == "owns"


def test_a_refusal_is_reported_separately_from_silence(index):
    """The gate must distinguish "I refuse to annotate this" from "nothing to map".

    Both used to reach the store as an empty ``ontology_property``, and
    ``write_relation`` coalesces — first non-null wins. The same edge is written
    once per chunk that asserts it, and those chunks disagree about the object's
    type, so one permissive chunk overrode every refusing one.

    Measured on conv-30: ``(Jon, HAS, students)`` is written five times. Four
    chunks type ``students`` PERSON and refuse; one types it POSSESSION and
    admits. Under plain coalesce the single admit won and the graph published
    ``Jon owns students`` — through TWO full re-ingests with the gate in place.
    """
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )

    # A person object: the predicate maps, and the gate refuses it.
    assert handler._map_predicate("HAS", "PERSON") == ("", 0.0)
    assert handler._range_blocked("HAS", "PERSON") is True

    # A possession object: mapped and admitted -> no refusal to record.
    assert handler._map_predicate("HAS", "POSSESSION")[0] == "owns"
    assert handler._range_blocked("HAS", "POSSESSION") is False

    # An unmappable predicate is SILENCE, not a refusal — it must not become
    # sticky, or one unmappable chunk would suppress every later annotation.
    # (map_predicate still reports the below-threshold similarity alongside the
    # empty label, so assert on the label rather than the pair.)
    assert handler._map_predicate("photosynthesizes", "PERSON")[0] == ""
    assert handler._range_blocked("photosynthesizes", "PERSON") is False


def test_no_ontology_configured_never_blocks(index):
    """Without an ontology there is no gate, so nothing is ever refused."""
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(store=None, extractor=None, embed_fn=None, ontology_index=None)
    assert handler._range_blocked("HAS", "PERSON") is False
