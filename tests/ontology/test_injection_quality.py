"""Ontology injection quality — does definition-matching surface the RIGHT terms?

``test_ontology_terms`` covers the mechanics (facets stay separate, top-k is
bounded, a term self-matches). This file covers the semantics: given ordinary
conversational prose, does the ontology term a human would pick actually rank
above one they would not?

Assertions are deliberately **pairwise and wide-margin**. Absolute cosines are
low and model-dependent (the catalog notes top-1 class medians ~0.20), and rank
positions shift between embedding models, so "works_as beats dislikes for a
sentence about a job" is a claim any competent embedder should satisfy whereas
"works_as ranks first" is not.

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
    """Why ranking weakness is survivable for a small ontology.

    Injection is gated by top-k alone (``DEFAULT_MIN_SIM`` is 0.0) and
    ``DEFAULT_TOP_K`` is 12, so for an ontology this size essentially every term
    reaches the extractor no matter how it ranks. Ranking only becomes
    load-bearing for ontologies larger than the budget — which is exactly where
    a regression here would start to bite.
    """
    from graphknows.ontology.catalog import DEFAULT_MIN_SIM, DEFAULT_TOP_K

    assert DEFAULT_MIN_SIM == 0.0, "a non-zero floor would make ranking load-bearing"

    emb = embed_one("Some ordinary sentence about nothing in particular.")
    classes = index.match_classes(emb, top_k=DEFAULT_TOP_K)
    properties = index.match_properties(emb, top_k=DEFAULT_TOP_K)

    assert len(classes) == len(index.classes), "every class should survive the budget"
    assert len(properties) == min(DEFAULT_TOP_K, len(index.properties))


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
