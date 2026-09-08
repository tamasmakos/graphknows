"""Only ``Inheritance`` may become a taxonomy edge (plan §11.7).

The defect this pins: ``_broader_up`` used to persist ``Inheritance``, ``Using``
and ``Perspective_on`` alike, so 683 of the hierarchy's 1464 edges — 47% — were
not subsumption, and the narrower-expansion walk handed siblings and background
frames back as specialisations.
"""

from __future__ import annotations

import pytest

from graphknows.symbolic.framenet import relations
from graphknows.symbolic.framenet.framenet import framenet

pytestmark = pytest.mark.unit


def test_inheritance_is_the_only_taxonomic_relation() -> None:
    assert {"Inheritance"} == relations.TAXONOMIC
    assert not (relations.TAXONOMIC & relations.NON_TAXONOMIC)


def test_using_is_not_taxonomy() -> None:
    """Shopping Uses Commerce_buy. Shopping is not a kind of buying — you can shop
    without buying anything, which is exactly why presupposition is not is-a."""
    assert "Commerce_buy" not in relations.broader_pairs(framenet()).get("Shopping", set())
    assert ("Shopping", "Using", "Commerce_buy") in relations.sym_rel_triples(framenet())


def test_perspective_on_is_not_taxonomy() -> None:
    """Commerce_buy perspectivises Commerce_goods-transfer — a sibling view, not a parent."""
    assert "Commerce_goods-transfer" not in relations.broader_pairs(framenet()).get(
        "Commerce_buy", set()
    )
    assert (
        "Commerce_buy",
        "Perspective_on",
        "Commerce_goods-transfer",
    ) in relations.sym_rel_triples(framenet())


def test_the_hierarchy_survives_the_narrowing() -> None:
    """Removing the wrong parents must not empty the taxonomy."""
    up = relations.broader_pairs(framenet())
    assert len(up) > 500, "Inheritance alone should still cover most frames"
    # Both fixture frames keep a genuine Inheritance parent.
    assert relations.broader_pairs(framenet())["Speed_description"] == {"Measurable_attributes"}
    assert "Getting" in relations.broader_pairs(framenet())["Commerce_buy"]


def test_every_shipped_relation_type_is_classified() -> None:
    """A new FrameNet relation type must surface, not be swept into a default.

    This guard found ``Metaphor`` on its first run — a tenth relation type absent
    from the canonical nine-relation lists.
    """
    assert relations.unknown_relation_types(framenet()) == set()


def test_expansion_may_still_follow_the_non_taxonomic_relations() -> None:
    """Widening a query is a different question from storing a claim."""
    from graphknows.symbolic.framenet.framenet import ontology_up

    assert "Commerce_buy" in ontology_up().get("Shopping", set())
