"""The syntactic-head merge layer: one referent, one node.

``_standard_key`` strips a determiner and singularizes, which cannot reach
"ideal dance studio" vs "dance studio" — they differ by an adjective. Measured on
a conv-30 run, one dance studio was split across six surfaces, and the
duplicates produced self-referential edges that read as structure while carrying
nothing. Multi-hop quality depends on nodes being merged: a path through a
referent split six ways is six short paths that each dead-end.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.consolidation.entity_resolution import _head_key
from graphknows.settings import GraphKnowsSettings


@pytest.fixture(scope="module")
def nlp() -> object:
    import spacy

    return spacy.load(GraphKnowsSettings().spacy_model)


@pytest.mark.parametrize(
    "surface",
    [
        "dance studio",
        "dance studios",
        "ideal dance studio",
        "the ideal dance studio by the water",
    ],
)
def test_descriptive_variants_share_one_key(surface: str, nlp: object) -> None:
    """Adjectives and trailing PPs describe; the head and its compounds name."""
    assert _head_key(surface, nlp) == "dance studio"


def test_a_possessor_is_dropped_from_the_name(nlp: object) -> None:
    """The owner belongs in a relation — keeping it splits one store per speaker."""
    assert _head_key("Jon's new store", nlp) == "store"


def test_a_proper_noun_head_is_never_reduced(nlp: object) -> None:
    """ "Door Dash" names a thing; reducing it would invent a common noun."""
    assert _head_key("Door Dash", nlp) == ""


def test_a_compound_is_not_reduced_to_its_head(nlp: object) -> None:
    """ "clothing store" is not "store" — the compound is part of the name."""
    assert _head_key("clothing store", nlp) == "clothing store"


def test_a_verb_phrase_yields_no_key(nlp: object) -> None:
    """A clause fragment is not a mention; refusing to key it leaves it alone."""
    assert _head_key("lost my job", nlp) == ""
