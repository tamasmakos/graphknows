"""Memory.corpus_ingest drives the golden-layer ingest + flush path.

Regression guard: this method used to delegate to ``CorpusIngestWorkflow``,
which was still wired for the retired stm/vector/ltm store split and called
``STMService`` with keyword arguments that no longer existed. Every call raised
a TypeError that the facade swallowed into ``{"error": ...,
"documents_processed": 0}``. The existing corpus tests exercised the workflow
class directly with mocks, so nothing covered the path an MCP client takes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory import Memory


def _memory() -> Memory:
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem.ingest_memory = AsyncMock(
        return_value={"file_id": "f", "chunks": 2, "entities": 3, "elapsed_s": 0.1, "errors": []}
    )
    mem.flush = AsyncMock(return_value={"unread_types": [], "errors": []})
    return mem


@pytest.mark.asyncio
async def test_ingests_every_document_then_consolidates_once() -> None:
    mem = _memory()
    out = await mem.corpus_ingest(
        [{"id": "d1", "text": "one"}, {"id": "d2", "text": "two"}], session_id="s1"
    )

    assert out["documents_processed"] == 2
    assert out["chunks_written"] == 4
    assert out["entities_extracted"] == 6
    assert out["errors"] == []
    assert "error" not in out
    # One shared session, reported on exactly once at the end.
    assert {c.kwargs["session_id"] for c in mem.ingest_memory.await_args_list} == {"s1"}
    mem.flush.assert_awaited_once_with()
    assert out["unread_types"] == []


@pytest.mark.asyncio
async def test_document_without_text_is_reported_not_fatal() -> None:
    mem = _memory()
    out = await mem.corpus_ingest([{"id": "d1", "text": ""}, {"id": "d2", "text": "ok"}])

    assert out["documents_processed"] == 1
    assert any("d1" in e for e in out["errors"])


@pytest.mark.asyncio
async def test_per_document_failure_does_not_abort_the_batch() -> None:
    mem = _memory()
    mem.ingest_memory = AsyncMock(
        side_effect=[
            RuntimeError("boom"),
            {"file_id": "f", "chunks": 1, "entities": 1, "elapsed_s": 0.1, "errors": []},
        ]
    )
    out = await mem.corpus_ingest([{"id": "d1", "text": "a"}, {"id": "d2", "text": "b"}])

    assert out["documents_processed"] == 1
    assert any("d1" in e and "boom" in e for e in out["errors"])
    mem.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_batch_short_circuits() -> None:
    mem = _memory()
    out = await mem.corpus_ingest([])

    assert out["documents_processed"] == 0
    mem.flush.assert_not_awaited()
