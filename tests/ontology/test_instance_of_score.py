"""``ENTITY -[:INSTANCE_OF {score}]-> ONTOLOGY_CLASS`` must rank the ENTITY.

The score on an INSTANCE_OF edge answers "how strongly is THIS entity an
instance of THAT class". A score that is a property of the (entity type,
ontology class) pair alone is identical for every entity sharing the type, so
ordering a class's members by it is ordering by a constant — it can rank
nothing, and a retrieval step that trusts it gets no signal at all. The
per-entity number the pipeline already carries is the extractor's confidence
(``score``, or ``confidence``), the same value that becomes ENTITY.confidence.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.symbolic.ontology import OntologyTerm


class _StubIndex:
    """The attributes ``_STMIngestHandler`` reads off an OntologyIndex.

    Not a real ``OntologyIndex``: building one embeds every term, and nothing
    here is about embeddings — "Person" is an exact class label, which
    ``_map_to_class`` resolves by lookup.
    """

    def __init__(self, classes: list[OntologyTerm]) -> None:
        self.classes = classes
        self.properties: list[OntologyTerm] = []

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]


class _FakeStore:
    """Records the INSTANCE_OF mappings instead of writing them."""

    def __init__(self) -> None:
        self.instance_of: list[dict[str, Any]] = []

    async def write_class_matches(self, chunk_id: str, matches: Any) -> None:
        return None

    async def write_groundings(self, rows: list[dict[str, Any]]) -> None:
        self.instance_of.extend(rows)

    def scores(self) -> dict[str, float]:
        return {m["name"]: m["score"] for m in self.instance_of}


def _handler(store: _FakeStore) -> _STMIngestHandler:
    stub = _StubIndex([OntologyTerm(uri="u:Person", label="Person", definition="", kind="class")])
    return _STMIngestHandler(
        store=store, extractor=None, embed_fn=None, ontology_index=stub, relation_index=stub
    )


@pytest.mark.asyncio
async def test_two_entities_of_one_class_carry_different_instance_of_scores() -> None:
    """Ada (0.95) and Bob (0.42) are both Persons, and Ada is the stronger one."""
    store = _FakeStore()
    await _handler(store)._link_ontology(
        "c1",
        [
            {"name": "Ada", "type": "Person", "score": 0.95},
            {"name": "Bob", "type": "Person", "score": 0.42},
        ],
        [],
    )

    assert {m["uri"] for m in store.instance_of} == {"u:Person"}

    scores = store.scores()
    assert len(set(scores.values())) == 2, f"one class, one number — cannot rank: {scores}"
    assert scores["Ada"] > scores["Bob"]
    # Exact label hit: the type->class factor is 1.0, so the edge carries the
    # entity's own confidence.
    assert scores["Ada"] == pytest.approx(0.95)
    assert scores["Bob"] == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_confidence_is_read_under_either_key_and_defaults_to_one() -> None:
    """``confidence`` is the key ``write_entities_bulk`` prefers; absent means 1.0."""
    store = _FakeStore()
    await _handler(store)._link_ontology(
        "c2",
        [
            {"name": "Cy", "type": "Person", "confidence": 0.6},
            {"name": "Di", "type": "Person"},
        ],
        [],
    )

    assert store.scores() == {"Cy": pytest.approx(0.6), "Di": pytest.approx(1.0)}
