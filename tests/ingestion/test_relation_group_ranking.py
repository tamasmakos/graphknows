"""Within a (domain, range) group, the property offered first is the best fit.

``_relation_labels`` groups the domain/range-satisfying properties by their
endpoint signature and round-robins one per group so nine interpersonal
Person->Person properties cannot take every slot. Which one it took per group was
whichever the vocabulary listed first — so ``is aunt of`` won the Person->Person
slot on a conversation about friends, and ``uses`` won over ``has output`` for a
painter and her painting (both recorded in the ontology_kg prototype).

Ranking INSIDE a group is ordering, never filtering: the offered set is identical,
only its order changes. Ranking ACROSS the whole vocabulary is the separate,
measured failure documented on ``_label_hints`` and is not what this does.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.symbolic.ontology.loader import OntologyTerm

# Orthogonal unit vectors, so each property's "fit" to a chunk is unambiguous.
_VECTORS = {
    "u:is aunt of": [1.0, 0.0, 0.0],
    "u:knows": [0.0, 1.0, 0.0],
    "u:has friend": [0.0, 0.0, 1.0],
    "u:worksFor": [0.0, 1.0, 0.0],
}
KNOWS_CHUNK = [0.0, 1.0, 0.0]


class _StubIndex:
    def __init__(self, classes: list[OntologyTerm], properties: list[OntologyTerm]) -> None:
        self.classes = classes
        self.properties = properties

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]

    def term_embedding(self, uri: str) -> list[float]:
        return _VECTORS.get(uri, [])


def _cls(label: str) -> OntologyTerm:
    return OntologyTerm(uri=f"u:{label}", label=label, definition="", kind="class")


def _prop(label: str, domain: tuple[str, ...], rng: tuple[str, ...]) -> OntologyTerm:
    return OntologyTerm(
        uri=f"u:{label}", label=label, definition="", kind="property", domain=domain, range=rng
    )


@pytest.fixture
def handler() -> _STMIngestHandler:
    # Insertion order puts the WRONG property first on purpose: that order is
    # what the unranked path returns, and is the defect under test.
    stub = _StubIndex(
        [_cls("Person"), _cls("Organization")],
        [
            _prop("is aunt of", ("Person",), ("Person",)),
            _prop("knows", ("Person",), ("Person",)),
            _prop("has friend", ("Person",), ("Person",)),
            _prop("worksFor", ("Person",), ("Organization",)),
        ],
    )
    return _STMIngestHandler(
        store=None, extractor=None, embed_fn=None, ontology_index=stub, relation_index=stub
    )


ENTITIES = [{"type": "Person"}, {"type": "Organization"}]


def test_unranked_the_first_listed_property_wins_its_group(handler) -> None:
    # The behaviour being replaced, pinned so the change is visible.
    offered = handler._relation_labels(ENTITIES)
    assert next(iter(offered)) == "is aunt of"


def test_ranking_puts_the_best_fitting_property_first_in_its_group(
    handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRAPHKNOWS_RELATION_GROUP_RANKING", "1")

    offered = handler._relation_labels(ENTITIES, KNOWS_CHUNK)

    assert next(iter(offered)) == "knows"


def test_ranking_reorders_but_never_changes_the_offered_set(
    handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = handler._relation_labels(ENTITIES)
    monkeypatch.setenv("GRAPHKNOWS_RELATION_GROUP_RANKING", "1")
    ranked = handler._relation_labels(ENTITIES, KNOWS_CHUNK)

    assert set(ranked) == set(plain)


def test_round_robin_across_groups_survives_the_reordering(
    handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One property per (domain, range) group before any group's second."""
    monkeypatch.setenv("GRAPHKNOWS_RELATION_GROUP_RANKING", "1")

    offered = list(handler._relation_labels(ENTITIES, KNOWS_CHUNK))

    # Person->Organization has exactly one member, so it must appear second —
    # before the Person->Person group's remaining two.
    assert offered[:2] == ["knows", "worksFor"]


def test_a_missing_embedding_leaves_the_order_untouched(
    handler, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tests and the lexical path both call without one; it must not reorder or raise.
    monkeypatch.setenv("GRAPHKNOWS_RELATION_GROUP_RANKING", "1")
    assert list(handler._relation_labels(ENTITIES, [])) == list(handler._relation_labels(ENTITIES))


def test_the_flag_is_off_by_default_so_the_embedding_is_ignored(handler) -> None:
    assert next(iter(handler._relation_labels(ENTITIES, KNOWS_CHUNK))) == "is aunt of"
