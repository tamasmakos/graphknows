"""Integration test: DETRetriever returns non-empty results against live ArcadeDB.

Requires ArcadeDB to be reachable. Skips automatically in environments where it
is not (controlled by the ``integration`` pytest marker and infrastructure checks).

This test specifically guards against the regression where CREATE PROPERTY E.*
DDL in _SCHEMA_DDL caused ensure_schema() to raise, preventing _get_retriever()
from ever returning a usable retriever and making every memory_query return empty.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.config import GraphKnowsSettings
    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/ltm",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ltm_ensure_schema_does_not_raise() -> None:
    """ensure_schema() on a live ArcadeDB must succeed without raising.

    This guards the regression where CREATE PROPERTY E.valid_at (and three
    sibling E.* properties) caused SchemaException "Type with name 'E' was not
    found", which made _get_retriever() fail on every call and forced every
    memory_query to return empty results.
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.config import GraphKnowsSettings
    from graphknows.config.builders.stores import build_arcadedb_ltm_store

    s = GraphKnowsSettings()
    store = build_arcadedb_ltm_store(s)
    await store._c.connect()
    try:
        # Must not raise — previously blew up on CREATE PROPERTY E.valid_at
        await store.ensure_schema()
    finally:
        await store._c.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_det_retriever_returns_results_after_ingest() -> None:
    """DETRetriever.retrieve() returns at least one passage for an ingested session.

    Writes a single chunk directly to the STM vector store so the semantic ANN
    branch can return it, then runs retrieve() and asserts non-empty fused_results.
    Cleans up after itself so the test is idempotent.
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.config import GraphKnowsSettings
    from graphknows.config.builders.stores import (
        build_arcadedb_ltm_store,
        build_arcadedb_stm_store,
        build_arcadedb_vector_store,
    )
    from graphknows.memory.query.retriever import DETRetriever
    from graphknows.memory.stores.embedder import embed_one

    s = GraphKnowsSettings()
    pg = build_arcadedb_vector_store(s)
    neo = build_arcadedb_ltm_store(s)
    mg = build_arcadedb_stm_store(s)
    await asyncio.gather(pg.connect(), neo._c.connect(), mg._c.connect())
    await neo.ensure_schema()

    test_session = f"test-det-retriever-integration-{uuid.uuid4().hex[:8]}"
    chunk_id = f"{test_session}-chunk-0"
    test_text = "Caroline went on vacation to Japan in May."

    # Write one chunk with embedding so the semantic ANN branch can find it
    emb = embed_one(test_text, s.embed_model)
    await pg.write_stm_chunk({
        "id": chunk_id,
        "session_id": test_session,
        "file_id": f"{test_session}-file",
        "text": test_text,
        "heading_path": "",
        "metadata": {},
        "embedding": emb,
    })

    try:
        retriever = DETRetriever(pg=pg, neo=neo, mg=mg, embed_model=s.embed_model)
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
        await pg.clear_stm(test_session)
        await pg.close()
