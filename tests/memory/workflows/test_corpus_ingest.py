"""Unit tests for CorpusIngestWorkflow — all mocked, no live backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.workflows.corpus_ingest import (
    CorpusIngestRequest,
    CorpusIngestWorkflow,
)


def _make_workflow() -> CorpusIngestWorkflow:
    return CorpusIngestWorkflow(
        mg_store=MagicMock(),
        pg_store=MagicMock(),
        neo_store=MagicMock(),
        extractor=MagicMock(),
        embed_fn=MagicMock(),
        openrouter_key="",
        rel_extractor=None,
    )


def _make_ingest_result(entities: int = 3, chunks: int = 2) -> MagicMock:
    r = MagicMock()
    r.entities = entities
    r.chunks = chunks
    return r


def _make_flush_result() -> MagicMock:
    r = MagicMock()
    r.nodes_written = 5
    r.edges_written = 4
    r.errors = []
    return r


@pytest.mark.asyncio
async def test_workflow_calls_stm_then_flush() -> None:
    """STMService.ingest called once per doc; SessionFlusher.flush called once."""
    docs = [
        {"id": "d1", "text": "Document one text.", "title": "Doc 1"},
        {"id": "d2", "text": "Document two text.", "title": "Doc 2"},
    ]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(return_value=_make_ingest_result())

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=_make_flush_result())

    with (
        patch(
            "graphknows.memory.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        result = await wf.run(request)

    assert mock_stm.ingest.call_count == 2
    assert mock_flusher.flush.call_count == 1
    assert result.documents_processed == 2
    assert result.nodes_written == 5
    assert result.errors == []


@pytest.mark.asyncio
async def test_workflow_skips_docs_without_text() -> None:
    """Document missing 'text' is skipped; error is recorded; STM not called for it."""
    docs = [
        {"id": "good", "text": "Has content."},
        {"id": "bad"},  # no 'text'
    ]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(return_value=_make_ingest_result())

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=_make_flush_result())

    with (
        patch(
            "graphknows.memory.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        result = await wf.run(request)

    assert mock_stm.ingest.call_count == 1
    assert result.documents_processed == 1
    assert len(result.errors) == 1
    assert "bad" in result.errors[0]


@pytest.mark.asyncio
async def test_workflow_stm_failure_collected_in_errors() -> None:
    """STM raise → error recorded; flush still attempted."""
    docs = [{"id": "d1", "text": "Some text."}]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(side_effect=RuntimeError("memgraph down"))

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=_make_flush_result())

    with (
        patch(
            "graphknows.memory.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        result = await wf.run(request)

    assert result.documents_processed == 0
    assert any("memgraph down" in e for e in result.errors)
    assert mock_flusher.flush.call_count == 1


@pytest.mark.asyncio
async def test_workflow_flush_failure_collected_in_errors() -> None:
    """Flush raise → error recorded in result."""
    docs = [{"id": "d1", "text": "Some text."}]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(return_value=_make_ingest_result())

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(side_effect=RuntimeError("neo4j unavailable"))

    with (
        patch(
            "graphknows.memory.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        result = await wf.run(request)

    assert any("neo4j unavailable" in e for e in result.errors)


@pytest.mark.asyncio
async def test_workflow_accepts_explicit_session_id() -> None:
    """Caller-supplied session_id is preserved in result."""
    explicit_sid = "my-idempotent-session-123"
    docs = [{"id": "d1", "text": "Content here."}]
    request = CorpusIngestRequest(documents=docs, session_id=explicit_sid)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(return_value=_make_ingest_result())

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=_make_flush_result())

    with (
        patch(
            "graphknows.memory.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        result = await wf.run(request)

    assert result.session_id == explicit_sid
    # Verify flush was called with the explicit session_id
    call_args = mock_flusher.flush.call_args
    assert call_args[0][0] == explicit_sid or call_args.args[0] == explicit_sid
