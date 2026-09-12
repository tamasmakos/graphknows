"""T020/FR-008: the golden-layer writers, and the evidence a fact cannot skip.

``graph_store.py`` wrote every vertex family; the writers split it into one
class per family. These pin what each writer emits — mocked client, no
ArcadeDB — and the invariant that gives the split its point: a fact with no
``ASSERTED_IN`` evidence is rejected at write, not repaired later.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.exceptions import StoreError
from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb.writers import (
    EntityWrite,
    EntityWriter,
    Evidence,
    FactWriter,
    SegmentWriter,
    SourceWriter,
    SymbolWriter,
)


@pytest.fixture
def store() -> ArcadeStoreBase:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return ArcadeStoreBase(client, "db")


def _statements(store: ArcadeStoreBase) -> list[str]:
    """The statement text of every command the store issued, in order."""
    return [call.args[1] for call in store.client.command.await_args_list]


def _params(store: ArcadeStoreBase, index: int) -> dict:
    return store.client.command.await_args_list[index].kwargs["params"]


def _fact() -> Fact:
    return Fact(id="f1", subject="e1", predicate="p:works_at", object="e2")


async def test_source_writer_merges_the_source_vertex(store: ArcadeStoreBase) -> None:
    source = Source(uri="file:///notes.md", content_hash="abc", mime="text/markdown")
    assert await SourceWriter(store).write(source) == source.id
    assert "UPDATE SOURCE SET" in _statements(store)[0]
    assert "UPSERT WHERE id = :id" in _statements(store)[0]
    assert _params(store, 0)["content_hash"] == "abc"


async def test_segment_writer_attaches_the_segment_to_its_source(
    store: ArcadeStoreBase,
) -> None:
    segment = Segment(
        source_id="s1", text="Mel works at Acme.", kind=SegmentKind.prose, byte_range=(0, 18)
    )
    assert await SegmentWriter(store).write(segment, [0.5, 0.25]) == segment.id
    vertex, edge = _statements(store)
    assert "MERGE (g:SEGMENT {id: $id})" in vertex
    assert "g.embedding = [0.5,0.25]" in vertex
    assert "MERGE (g)-[:PART_OF]->(s)" in edge
    assert _params(store, 0)["byte_end"] == 18


async def test_segment_writer_leaves_an_empty_embedding_unset(store: ArcadeStoreBase) -> None:
    """A zero-length vector is refused by the index, so none is sent."""
    segment = Segment(source_id="s1", text="x", kind=SegmentKind.prose, byte_range=(0, 1))
    await SegmentWriter(store).write(segment)
    assert "embedding" not in _statements(store)[0]


async def test_segment_writer_links_consecutive_segments(store: ArcadeStoreBase) -> None:
    await SegmentWriter(store).link_next("g1", "g2")
    assert "MERGE (a)-[:NEXT]->(b)" in _statements(store)[0]
    assert _params(store, 0) == {"previous_id": "g1", "segment_id": "g2"}


async def test_entity_writer_writes_the_entity_and_its_mention(
    store: ArcadeStoreBase,
) -> None:
    writer = EntityWriter(store)
    await writer.write(
        EntityWrite(
            id="e1", name="Acme", name_norm="acme", embedding=[0.5], type_histogram={"org": 1}
        )
    )
    await writer.write_mention(
        Mention(segment_id="g1", entity_id="e1", surface="Acme", span=(13, 17))
    )
    entity, mention = _statements(store)
    assert "UPDATE ENTITY SET" in entity and "UPSERT WHERE id = :id" in entity
    assert "embedding = [0.5]" in entity
    assert _params(store, 0)["type_histogram"] == {"org": 1}
    assert "MENTIONS" in mention
    assert _params(store, 1)["span"] == "13:17"


async def test_entity_writer_accumulates_the_type_histogram(store: ArcadeStoreBase) -> None:
    """A second read under a label adds to the tally instead of resetting it."""
    store.client.query = AsyncMock(return_value=[{"histogram": {"org": 2, "place": 1}}])
    await EntityWriter(store).write(
        EntityWrite(id="e1", name="Acme", name_norm="acme", type_histogram={"org": 1})
    )
    assert _params(store, 0)["type_histogram"] == {"org": 3, "place": 1}
    assert "embedding" not in _statements(store)[0]


async def test_symbol_writer_writes_concepts_with_their_embedding(
    store: ArcadeStoreBase,
) -> None:
    writer = SymbolWriter(store)
    concept = ConceptRef(uri="c:company", label="Company", definition="a firm", pack="core")
    predicate = PredicateRef(
        id="p:works_at",
        label="works at",
        definition="employment",
        canonical="works_at",
        functional=True,
        pack="core",
    )
    assert await writer.write_concept(concept, [0.1, 0.2]) == concept.uri
    assert await writer.write_predicate(predicate) == predicate.id
    assert "c.embedding = [0.1,0.2]" in _statements(store)[0]
    assert "embedding" not in _params(store, 0)
    assert _params(store, 1)["functional"] is True


async def test_fact_writer_links_the_triple_and_its_evidence(store: ArcadeStoreBase) -> None:
    fact = _fact()
    written = await FactWriter(store).write(fact, [Evidence(segment_id="g1", span=(0, 18))])
    assert written == "f1"
    vertex, triple, evidence = _statements(store)
    assert "MERGE (f:FACT {id: $id})" in vertex
    assert "SUBJECT_OF" in triple and "OBJECT_IS" in triple and "USES" in triple
    assert "MERGE (f)-[a:ASSERTED_IN]->(g)" in evidence
    assert _params(store, 2)["span"] == "0:18"


async def test_fact_without_evidence_is_rejected_at_write(store: ArcadeStoreBase) -> None:
    """FR-008: no ASSERTED_IN edge, no fact — and nothing written on the way out."""
    with pytest.raises(StoreError, match="ASSERTED_IN"):
        await FactWriter(store).write(_fact(), [])
    store.client.command.assert_not_awaited()
