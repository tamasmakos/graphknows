"""Integration test: DETRetriever returns non-empty results against live ArcadeDB.

Requires ArcadeDB to be reachable. Skips automatically in environments where it
is not (controlled by the ``integration`` pytest marker and infrastructure checks).

This test specifically guards against the regression where CREATE PROPERTY E.*
DDL in _SCHEMA_DDL caused ensure_schema() to raise, preventing _get_retriever()
from ever returning a usable retriever and making every memory_query return empty.
"""

from __future__ import annotations

import uuid

import pytest


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/mem",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_store_ensure_schema_does_not_raise() -> None:
    """ensure_schema(dims) on a live ArcadeDB must succeed without raising.

    This guards the regression where CREATE PROPERTY E.valid_at (and three
    sibling E.* properties) caused SchemaException "Type with name 'E' was not
    found", which made _get_retriever() fail on every call and forced every
    memory_query to return empty results.
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_graph_store
    from graphknows.storage.embedder import embed_dim

    s = GraphKnowsSettings()
    store = build_graph_store(s)
    await store.connect()
    try:
        # Must not raise — previously blew up on CREATE PROPERTY E.valid_at
        await store.ensure_schema(s.embed_dimensions or embed_dim())
    finally:
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_det_retriever_returns_results_after_ingest() -> None:
    """DETRetriever.retrieve() returns at least one passage for an ingested session.

    Writes a single chunk with its embedding to the graph store so the semantic
    ANN branch can return it, then runs retrieve() and asserts non-empty
    fused_results. Cleans up after itself so the test is idempotent.
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.retrieval.retriever import DETRetriever
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_graph_store
    from graphknows.storage.embedder import embed_dim, embed_one

    s = GraphKnowsSettings()
    store = build_graph_store(s)
    await store.connect()
    await store.ensure_schema(s.embed_dimensions or embed_dim())

    test_session = f"test-det-retriever-integration-{uuid.uuid4().hex[:8]}"
    chunk_id = f"{test_session}-chunk-0"
    test_text = "Caroline went on vacation to Japan in May."

    # Write one chunk with embedding so the semantic ANN branch can find it
    await store.write_chunk(
        {
            "chunk_id": chunk_id,
            "session_id": test_session,
            "file_id": f"{test_session}-file",
            "text": test_text,
            "embedding": embed_one(test_text, s.embed_model),
        }
    )

    try:
        retriever = DETRetriever(store=store, embed_model=s.embed_model)
        ctx = await retriever.retrieve(
            "Where did Caroline go on vacation?",
            session_id=test_session,
            top_k=10,
            scope="stm",
        )

        assert len(ctx.fused_results) > 0, (
            "DETRetriever returned 0 results for a freshly-ingested chunk. "
            "Check that ensure_schema passes and the semantic ANN branch is working."
        )
        assert ctx.search_type == "DET"
        assert all("chunk_id" in r and "text" in r and "rrf_score" in r for r in ctx.fused_results)
    finally:
        # Clean up the test session
        await store.delete_session(test_session)
        await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_entities_are_shared_across_relation_writes() -> None:
    """Two relation writes reusing an entity name must not raise on the MERGE.

    Regression for the DuplicatedKeyException where the old STM schema had a
    single-column UNIQUE index on ENTITY(name) while every write MERGEd on
    (name, session_id) — entity names were global while the merge key assumed
    per-session scoping, so a second session reusing a name (e.g. "Melanie")
    failed with ``DuplicatedKeyException``.

    GraphStore's golden-layer schema keys ENTITY on ``name_norm`` alone —
    globally shared by design, with no session_id in the merge key at all — so
    this now guards the simpler invariant the fix converged on: writing the
    same entity name from two independent ``write_relation`` calls never 409s
    and always MERGEs onto the same single ENTITY node.
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_graph_store
    from graphknows.storage.embedder import embed_dim

    s = GraphKnowsSettings()
    store = build_graph_store(s)
    await store.connect()
    await store.ensure_schema(s.embed_dimensions or embed_dim())

    suffix = uuid.uuid4().hex[:8]
    subject, obj = f"Melanie-{suffix}", f"camping-{suffix}"

    try:
        # Same (subject, object) names written twice — before the fix this
        # raised DuplicatedKeyException on the ENTITY unique index.
        for _ in range(2):
            await store.write_relation(subject, "LIKES", obj, fact="evidence", confidence=0.9)

        rows = await store.query(
            "MATCH (e:ENTITY {name_norm: $name}) RETURN count(e) AS n",
            name=subject.casefold(),
        )
        assert rows[0]["n"] == 1
    finally:
        await store.drop_entities([subject.casefold(), obj.casefold()])
        await store.close()
