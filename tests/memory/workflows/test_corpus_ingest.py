"""Unit tests for CorpusIngestWorkflow — all mocked, no live backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.ingestion.workflows.corpus_ingest import (
    CorpusIngestRequest,
    CorpusIngestWorkflow,
)


def _make_workflow() -> CorpusIngestWorkflow:
    return CorpusIngestWorkflow(
        stm_store=MagicMock(),
        vec_store=MagicMock(),
        ltm_store=MagicMock(),
        extractor=MagicMock(),
        embed_fn=MagicMock(),
        llm_api_key="",
        rel_extractor=None,
    )


def _make_ingest_result(entities: int = 3, chunks: int = 2) -> MagicMock:
    r = MagicMock()
    r.entity_count = entities
    r.chunk_count = chunks
    return r


def _make_promote_result() -> MagicMock:
    r = MagicMock()
    r.nodes_written = 5
    r.edges_written = 4
    r.errors = []
    return r


@pytest.mark.asyncio
async def test_workflow_calls_stm_then_flush() -> None:
    """STMService.ingest called once per doc; LTMPromotionPipeline.promote called once."""
    docs = [
        {"id": "d1", "text": "Document one text.", "title": "Doc 1"},
        {"id": "d2", "text": "Document two text.", "title": "Doc 2"},
    ]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(return_value=_make_ingest_result())

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=_make_promote_result())

    with (
        patch(
            "graphknows.ingestion.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline",
            return_value=mock_pipeline,
        ),
    ):
        result = await wf.run(request)

    assert mock_stm.ingest.call_count == 2
    assert mock_pipeline.promote.call_count == 1
    assert result.documents_processed == 2
    assert result.entities_extracted == 6
    assert result.chunks_written == 4
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
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(return_value=_make_ingest_result())

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=_make_promote_result())

    with (
        patch(
            "graphknows.ingestion.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline",
            return_value=mock_pipeline,
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
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(side_effect=RuntimeError("stm down"))

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=_make_promote_result())

    with (
        patch(
            "graphknows.ingestion.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline",
            return_value=mock_pipeline,
        ),
    ):
        result = await wf.run(request)

    assert result.documents_processed == 0
    assert any("stm down" in e for e in result.errors)
    assert mock_pipeline.promote.call_count == 1


@pytest.mark.asyncio
async def test_workflow_flush_failure_collected_in_errors() -> None:
    """Promote raise → error recorded in result."""
    docs = [{"id": "d1", "text": "Some text."}]
    request = CorpusIngestRequest(documents=docs)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(return_value=_make_ingest_result())

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(side_effect=RuntimeError("ltm unavailable"))

    with (
        patch(
            "graphknows.ingestion.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline",
            return_value=mock_pipeline,
        ),
    ):
        result = await wf.run(request)

    assert any("ltm unavailable" in e for e in result.errors)


@pytest.mark.asyncio
async def test_workflow_accepts_explicit_session_id() -> None:
    """Caller-supplied session_id is preserved in result."""
    explicit_sid = "my-idempotent-session-123"
    docs = [{"id": "d1", "text": "Content here."}]
    request = CorpusIngestRequest(documents=docs, session_id=explicit_sid)
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(return_value=_make_ingest_result())

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=_make_promote_result())

    with (
        patch(
            "graphknows.ingestion.stm.service.STMService",
            return_value=mock_stm,
        ),
        patch(
            "graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline",
            return_value=mock_pipeline,
        ),
    ):
        result = await wf.run(request)

    assert result.session_id == explicit_sid
    (promotion_request,), _ = mock_pipeline.promote.call_args
    assert promotion_request.session_id == explicit_sid


@pytest.mark.asyncio
async def test_corpus_ingest_connects_stm_on_caller_loop() -> None:
    """corpus_ingest connects the STM service on the caller's loop (await), not a bg loop."""
    wf = _make_workflow()

    mock_stm = MagicMock()
    mock_stm.connect = AsyncMock(return_value=None)
    mock_stm.ingest = AsyncMock(return_value=_make_ingest_result())

    mock_pipeline = MagicMock()
    mock_pipeline.promote = AsyncMock(return_value=_make_promote_result())

    with (
        patch("graphknows.ingestion.stm.service.STMService", return_value=mock_stm),
        patch("graphknows.ingestion.promotion.pipeline.LTMPromotionPipeline", return_value=mock_pipeline),
    ):
        request = CorpusIngestRequest(documents=[{"id": "d1", "text": "Test text."}])
        await wf.run(request)

    mock_stm.connect.assert_awaited_once()
