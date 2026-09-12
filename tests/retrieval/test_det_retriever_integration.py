"""Integration test: DETRetriever returns non-empty results against live ArcadeDB.

Requires ArcadeDB to be reachable. Skips automatically in environments where it
is not (controlled by the ``integration`` pytest marker and infrastructure checks).

Both tests run on a scratch ``mem_test_<uuid>`` database and drop it after:
the shared default database is whatever the developer last ingested, stamped
or not, and a test must neither depend on that nor delete from it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


@pytest.fixture
async def scratch_store(arcadedb_required: None) -> AsyncIterator[GraphStore]:
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client

    store = GraphStore(
        client=build_arcadedb_client(GraphKnowsSettings()), db=f"mem_test_{uuid.uuid4().hex[:8]}"
    )
    await store.connect()
    try:
        yield store
    finally:
        await store.drop_database()
        await store.close()


@pytest.mark.integration
async def test_graph_store_ensure_schema_does_not_raise(scratch_store: GraphStore) -> None:
    """ensure_schema(dims) on a live ArcadeDB must succeed without raising.

    This guards the regression where CREATE PROPERTY E.valid_at (and three
    sibling E.* properties) caused SchemaException "Type with name 'E' was not
    found", which made _get_retriever() fail on every call and forced every
    memory_query to return empty results.
    """
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage.embedder import embed_dim

    s = GraphKnowsSettings()
    await scratch_store.ensure_schema(s.embed_dimensions or embed_dim())


@pytest.mark.integration
async def test_det_retriever_returns_results_after_ingest(scratch_store: GraphStore) -> None:
    """DETRetriever.retrieve() returns at least one passage for an ingested session.

    Writes one session SOURCE with one SEGMENT carrying its embedding, so the
    semantic ANN branch can return it, then runs retrieve() scoped to that
    session and asserts non-empty fused_results. The SOURCE is written with
    raw Cypher so this test fails on the retriever alone, never on a writer.
    """
    from graphknows.models.segment import Segment, SegmentKind
    from graphknows.models.source import session_uri
    from graphknows.retrieval.retriever import DETRetriever
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage.arcadedb._sql import vector_literal
    from graphknows.storage.arcadedb.writers import SegmentWriter
    from graphknows.storage.embedder import embed_dim, embed_one

    s = GraphKnowsSettings()
    store = scratch_store
    await store.ensure_schema(s.embed_dimensions or embed_dim())

    test_session = f"test-det-retriever-integration-{uuid.uuid4().hex[:8]}"
    test_text = "Caroline went on vacation to Japan in May."
    await store.command(
        "MERGE (s:SOURCE {id: 'src-1'}) SET s.uri = $uri, s.content_hash = $hash, "
        "s.imported_at = '2026-01-01T00:00:00+00:00'",
        uri=session_uri(test_session),
        hash=test_session,
    )
    segment = Segment(
        source_id="src-1", text=test_text, kind=SegmentKind.turn, byte_range=(0, len(test_text))
    )
    await SegmentWriter(store).write(segment)
    embedding = vector_literal(embed_one(test_text, s.embed_model))
    await store.command(
        f"MATCH (g:SEGMENT {{id: $id}}) SET g.embedding = {embedding}",
        id=segment.id,
    )

    retriever = DETRetriever(store=store, embed_model=s.embed_model)
    ctx = await retriever.retrieve(
        "Where did Caroline go on vacation?", session_id=test_session, top_k=10
    )

    assert len(ctx.fused_results) > 0, (
        "DETRetriever returned 0 results for a freshly-ingested segment. "
        "Check that ensure_schema passes and the semantic ANN branch is working."
    )
    assert ctx.search_type == "DET"
    assert all("chunk_id" in r and "text" in r and "rrf_score" in r for r in ctx.fused_results)
    assert ctx.fused_results[0]["chunk_id"] == segment.id
