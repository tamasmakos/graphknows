"""Integration test: ``Memory.ingest`` then ``Memory.recall`` against live ArcadeDB.

The unit suites stub the store, so every defect that only exists in the SQL —
a MAP parameter the dialect rejects, an edge whose MATCH finds no vertex — is
invisible to them. This runs the real write path on a scratch
``mem_test_<uuid>`` database and reads back the graph it claims to write: the
symbols, the two labelling edges, the segment adjacency and the embeddings the
retriever needs, and finally a recall carrying evidence a caller can resolve.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from hashlib import sha256
from typing import Any

import pytest

from graphknows.memory import Memory
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source
from graphknows.packs.agent import AgentPack
from graphknows.packs.code import CodePack

_CODE = "def load_packs(packs):\n    return LoadedPacks(packs)\n"
_PROSE = "The team decided to ship the loader before the parser.\n"
_CONTENT = _CODE + _PROSE

_SOURCE = Source(
    uri="test://live/loader.py",
    content_hash=sha256(_CONTENT.encode()).hexdigest(),
    mime="text/plain",
    meta={"branch": "main"},
)

_SEGMENTS = (
    Segment(
        source_id=_SOURCE.id,
        text=_CODE,
        kind=SegmentKind.code,
        path="load_packs",
        byte_range=(0, len(_CODE.encode())),
    ),
    Segment(
        source_id=_SOURCE.id,
        text=_PROSE,
        kind=SegmentKind.prose,
        path="notes",
        byte_range=(len(_CODE.encode()), len(_CONTENT.encode())),
    ),
)


@pytest.fixture
async def ingested(arcadedb_required: None) -> AsyncIterator[Memory]:
    """A scratch namespace with one two-segment source ingested through both packs."""
    memory = Memory(namespace=f"test_{uuid.uuid4().hex[:8]}", packs=(CodePack(), AgentPack()))
    try:
        await memory.ingest(_SOURCE, _SEGMENTS)
        yield memory
    finally:
        await memory.drop_namespace()
        await memory.close()


async def _rows(memory: Memory, query: str, **params: Any) -> list[dict[str, Any]]:
    """Read the scratch namespace back through a store of our own."""
    from graphknows.settings import get_settings
    from graphknows.storage import build_graph_store

    store = build_graph_store(get_settings(), memory.namespace)
    await store.connect()
    try:
        return await store.query(query, **params)
    finally:
        await store.close()


@pytest.mark.integration
async def test_the_source_and_its_segments_round_trip(ingested: Memory) -> None:
    """The source keeps its meta — a MAP the Cypher dialect cannot bind — and its segments."""
    sources = await _rows(
        ingested, "MATCH (s:SOURCE {id: $id}) RETURN s.uri AS uri, s.meta AS meta", id=_SOURCE.id
    )
    segments = await _rows(
        ingested,
        "MATCH (g:SEGMENT)-[:PART_OF]->(s:SOURCE {id: $id}) "
        "RETURN g.id AS id, g.embedding IS NOT NULL AS embedded",
        id=_SOURCE.id,
    )

    assert [row["uri"] for row in sources] == [_SOURCE.uri]
    assert sources[0]["meta"] == {"branch": "main"}
    assert {row["id"] for row in segments} == {segment.id for segment in _SEGMENTS}
    assert all(row["embedded"] for row in segments)


@pytest.mark.integration
async def test_consecutive_segments_are_linked_by_next(ingested: Memory) -> None:
    """The retriever's neighbour walk reads NEXT, so the write path must lay it down."""
    rows = await _rows(
        ingested, "MATCH (a:SEGMENT)-[:NEXT]->(b:SEGMENT) RETURN a.id AS a, b.id AS b"
    )

    first, second = _SEGMENTS
    assert [(row["a"], row["b"]) for row in rows] == [(first.id, second.id)]


@pytest.mark.integration
async def test_entities_carry_their_type_histogram_and_embedding(ingested: Memory) -> None:
    """An entity without its surface embedding is invisible to the ANN candidate branch."""
    rows = await _rows(
        ingested,
        "MATCH (e:ENTITY) RETURN e.name AS name, e.type_histogram AS histogram, "
        "e.embedding IS NOT NULL AS embedded",
    )

    by_name = {row["name"]: row for row in rows}
    assert by_name["load_packs"]["histogram"] == {"function": 1}
    assert all(row["embedded"] for row in rows)


@pytest.mark.integration
async def test_concepts_are_indexed_by_the_embedding_of_their_definition(
    ingested: Memory,
) -> None:
    """A concept's embedding is its identity in the representation layer, not a side-car."""
    rows = await _rows(
        ingested,
        "MATCH (c:CONCEPT {uri: 'code:function'}) "
        "RETURN c.label AS label, c.embedding IS NOT NULL AS embedded",
    )

    assert [(row["label"], row["embedded"]) for row in rows] == [("function", True)]


@pytest.mark.integration
async def test_each_segment_evokes_the_concepts_its_mentions_named(ingested: Memory) -> None:
    """Bottom-up labelling (FR-002): both edges top-down activation reads back."""
    evoked = await _rows(
        ingested, "MATCH (:SEGMENT)-[:EVOKES]->(c:CONCEPT) RETURN DISTINCT c.uri AS uri"
    )
    instantiated = await _rows(
        ingested, "MATCH (:ENTITY)-[:INSTANCE_OF]->(c:CONCEPT) RETURN DISTINCT c.uri AS uri"
    )

    assert {"code:function", "agent:decision"} <= {row["uri"] for row in evoked}
    assert {"code:function", "agent:decision"} <= {row["uri"] for row in instantiated}


@pytest.mark.integration
async def test_recall_returns_a_fact_whose_evidence_resolves(ingested: Memory) -> None:
    """The atom of recall is a fact with evidence a caller can go and read (FR-008)."""
    result = await ingested.recall("which function calls what")

    assert result.facts, "no fact recalled for an ingested code symbol"
    spans = {segment.byte_range: segment.text for segment in _SEGMENTS}
    for recalled in result.facts:
        for evidence in recalled.evidence:
            assert evidence.source_uri == _SOURCE.uri
            assert spans[evidence.byte_range] == evidence.text
