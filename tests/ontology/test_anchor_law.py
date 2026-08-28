"""The anchor law: a grounding candidate may come ONLY from the lexical index.

A purely-cosine match against the class vocabulary is a retrieval tag, never a
persisted ``ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`` edge. Only a lexical hit
(an exact label match, the candidate stage the extractor's vocabulary was
built from) may become a grounding.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.symbolic.ontology import OntologyTerm


class _StubIndex:
    """Same shape as tests/ontology/test_instance_of_score.py's ``_StubIndex``,
    but its ``match_classes`` returns a confident cosine hit that is NOT in
    ``classes`` — i.e. the lexicon has no entry for the entity's type at all.
    """

    def __init__(self, classes: list[OntologyTerm]) -> None:
        self.classes = classes
        self.properties: list[OntologyTerm] = []

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]

    def match_classes(
        self, embedding: list[float], top_k: int = 1, min_sim: float = 0.0
    ) -> list[tuple[OntologyTerm, float]]:
        return [
            (
                OntologyTerm(uri="u:Vehicle", label="Vehicle", definition="", kind="class"),
                0.91,
            )
        ]


class _FakeStore:
    """Records groundings via the NEW write choke point, ``write_groundings``."""

    def __init__(self) -> None:
        self.groundings: list[dict[str, Any]] = []

    async def write_class_matches(self, chunk_id: str, matches: Any) -> None:
        return None

    async def write_groundings(self, rows: list[dict[str, Any]]) -> None:
        self.groundings.extend(rows)


def _handler(store: _FakeStore, classes: list[OntologyTerm]) -> _STMIngestHandler:
    stub = _StubIndex(classes)
    return _STMIngestHandler(
        store=store, extractor=None, embed_fn=None, ontology_index=stub, relation_index=stub
    )


@pytest.mark.asyncio
async def test_unanchored_cosine_match_never_becomes_a_grounding(monkeypatch: Any) -> None:
    """No class labelled 'quantum widget' exists in the lexicon: an empty
    lexical candidate set. Cosine alone (0.91 similarity to 'Vehicle') must
    NOT be written as a grounding edge.
    """
    monkeypatch.setattr("graphknows.storage.embedder.embed_one", lambda text: [0.1, 0.2, 0.3])
    store = _FakeStore()
    # Lexicon has no term whose label casefold-matches 'quantum widget'.
    handler = _handler(store, classes=[])

    await handler._link_ontology(
        "c1", [{"name": "Ada", "type": "quantum widget", "score": 0.9}], []
    )

    assert store.groundings == [], (
        f"a purely-cosine match must never be persisted as a grounding: {store.groundings}"
    )


@pytest.mark.asyncio
async def test_anchored_exact_label_match_is_still_written(monkeypatch: Any) -> None:
    """A lexical hit (exact label == entity type) is anchored and IS written."""
    monkeypatch.setattr("graphknows.storage.embedder.embed_one", lambda text: [0.1, 0.2, 0.3])
    store = _FakeStore()
    handler = _handler(
        store, classes=[OntologyTerm(uri="u:Person", label="Person", definition="", kind="class")]
    )

    await handler._link_ontology("c1", [{"name": "Ada", "type": "Person", "score": 0.9}], [])

    assert len(store.groundings) == 1
    assert store.groundings[0]["uri"] == "u:Person"
    assert store.groundings[0]["name"] == "Ada"
