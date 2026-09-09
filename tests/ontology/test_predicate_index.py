"""A relation is a predicate symbol, resolved through the index — never a free string."""

from __future__ import annotations

from graphknows.models import PredicateRef
from graphknows.symbolic.predicates import PredicateIndex


def _predicate(predicate_id: str, label: str) -> PredicateRef:
    return PredicateRef(
        id=predicate_id,
        label=label,
        definition=f"Test definition of {label}.",
        canonical=label,
        pack="demo",
    )


def test_a_label_resolves_to_the_predicate_symbol() -> None:
    index = PredicateIndex([_predicate("demo:works_at", "works at")])

    resolved = index.resolve("works at")

    assert resolved is not None and resolved.id == "demo:works_at"


def test_case_spacing_and_underscores_reach_one_symbol() -> None:
    index = PredicateIndex([_predicate("demo:works_at", "works at")])

    assert {index.resolve(surface) for surface in ("Works At", "works_at", "  WORKS  at ")} == {
        index.resolve("works at")
    }


def test_the_canonical_rendering_and_the_id_also_resolve() -> None:
    predicate = PredicateRef(
        id="demo:member_of",
        label="belongs to",
        definition="Membership of a group.",
        canonical="member of",
        pack="demo",
    )
    index = PredicateIndex([predicate])

    assert index.resolve("member of") is index.resolve("demo:member_of") is not None


def test_a_relation_the_pack_does_not_name_resolves_to_nothing() -> None:
    index = PredicateIndex([_predicate("demo:works_at", "works at")])

    assert index.resolve("founded") is None


def test_the_first_predicate_declaring_a_surface_keeps_it() -> None:
    index = PredicateIndex(
        [_predicate("demo:works_at", "works at"), _predicate("demo:worked_at", "works_at")]
    )

    resolved = index.resolve("works at")

    assert resolved is not None and resolved.id == "demo:works_at"


def test_a_predicate_carries_its_functional_flag_through_the_index() -> None:
    functional = PredicateRef(
        id="demo:decision_status",
        label="decision status",
        definition="The current status of a decision.",
        canonical="decision status",
        functional=True,
        pack="demo",
    )
    index = PredicateIndex([functional])

    resolved = index.resolve("decision status")

    assert resolved is not None and resolved.functional
