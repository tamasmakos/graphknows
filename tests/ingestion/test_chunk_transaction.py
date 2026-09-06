"""One chunk's writes are one transaction (FR-015, SC-008).

``_write_extracted`` issues a dozen commands per chunk — CHUNK, entities and
MENTIONS, class matches, senses, frames, frame instances, relations, temporal
facts. Until now each was its own auto-committed ArcadeDB statement, so a
failure halfway through left a *visible* CHUNK with no relations, no frames and
no dates: indistinguishable, to every reader, from a chunk that genuinely had
none. Silent corruption, not a visible error.

Under test:
  * committed  — a clean chunk is in the graph, as before
  * rolled back — a chunk that fails mid-write leaves nothing behind, not even
                  the CHUNK vertex written by the first command
  * counted    — the rollback reaches ``IngestResult.abstentions``; a rollback
                 nobody can count is the same defect wearing a different mask
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.storage.arcadedb.graph_store import GraphStore


class _StubExtractor:
    """Always finds the same one entity — the chunk just needs writes to make."""

    def extract(self, text: str, *args: Any, **kwargs: Any) -> ExtractionResult:
        return ExtractionResult(entities=[{"name": "Gina", "type": "PERSON", "score": 0.9}])


class _FailingStore(GraphStore):
    """A GraphStore whose second write raises, mid-chunk.

    ``write_chunk`` runs first and really writes; the entity write that follows
    it explodes. That is exactly the window this transaction closes.
    """

    async def write_entities_bulk(
        self, session_id: str, chunk_id: str, entities: list[dict[str, Any]]
    ) -> None:
        raise RuntimeError("mid-chunk failure")


async def _ingest(store: GraphStore, session_id: str) -> Any:
    from graphknows.ingestion.stm.ingest import _STMIngestHandler

    handler = _STMIngestHandler(
        store=store,
        extractor=_StubExtractor(),
        embed_fn=lambda _t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return await handler.ingest_text("Gina opened a clothing store.", session_id=session_id)


def _store(cls: type[GraphStore]) -> GraphStore:
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client

    settings = GraphKnowsSettings()
    return cls(client=build_arcadedb_client(settings), db=f"mem_test_{uuid.uuid4().hex[:8]}")


async def _chunk_count(store: GraphStore) -> int:
    rows = await store.query("MATCH (c:CHUNK) RETURN count(c) AS n")
    return int(rows[0]["n"]) if rows else 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_clean_chunk_commits(arcadedb_required: None) -> None:
    """The transaction is not a write barrier: an ordinary ingest still lands."""
    store = _store(GraphStore)
    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        result = await _ingest(store, "sess-tx-ok")

        assert result.errors == []
        assert await _chunk_count(store) == 1
        assert "chunk_write_rolled_back" not in result.abstentions
    finally:
        await store.drop_database()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_failed_chunk_rolls_back_and_is_counted(arcadedb_required: None) -> None:
    """The CHUNK written before the failure is gone, and the loss is reported."""
    store = _store(_FailingStore)
    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        result = await _ingest(store, "sess-tx-fail")

        assert await _chunk_count(store) == 0, "the pre-failure CHUNK survived the rollback"
        assert result.abstentions.get("chunk_write_rolled_back") == 1
        # The failure is still an error, not just an abstention: a rollback that
        # reported success would hide a chunk the file never got.
        assert result.errors, "a rolled-back chunk must still surface as an error"
    finally:
        await store.drop_database()
