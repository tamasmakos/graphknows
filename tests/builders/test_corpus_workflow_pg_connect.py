"""Test build_corpus_workflow pg.connect() behaviour after event-loop fix."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_build_corpus_workflow_does_not_connect_pg() -> None:
    """build_corpus_workflow must NOT call pg.connect() — connection deferred to run()."""
    mock_pg = MagicMock()
    mock_pg.connect = AsyncMock(return_value=None)

    with (
        patch("graphknows.config.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_pgvector_store", return_value=mock_pg),
        patch("graphknows.config.builders.stores.build_neo4j_store", return_value=MagicMock()),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor", MagicMock()),
        patch(
            "graphknows.memory.workflows.corpus_ingest.CorpusIngestWorkflow",
            return_value=MagicMock(),
        ),
        patch("graphknows.memory.stores.embedder.embed_one", MagicMock()),
    ):
        from graphknows.config.builders.stores import build_corpus_workflow
        from graphknows.config import GraphKnowsSettings

        settings = MagicMock(spec=GraphKnowsSettings)
        settings.openrouter_api_key = None
        settings.relation_extraction_mode = "gliner2"
        settings.taxonomy_file = None

        build_corpus_workflow(settings)

        # pg.connect() must NOT have been called during build
        mock_pg.connect.assert_not_called()


@pytest.mark.asyncio
async def test_workflow_run_connects_pg() -> None:
    """workflow.run() must call pg.connect() when _pool is not yet set."""
    from graphknows.memory.workflows.corpus_ingest import (
        CorpusIngestRequest,
        CorpusIngestWorkflow,
    )

    mock_pg = MagicMock()
    mock_pg._pool = None  # not yet connected
    mock_pg.connect = AsyncMock(return_value=None)

    mock_mg = MagicMock()
    mock_neo = MagicMock()
    mock_extractor = MagicMock()
    mock_embed_fn = MagicMock()

    workflow = CorpusIngestWorkflow(
        mg_store=mock_mg,
        pg_store=mock_pg,
        neo_store=mock_neo,
        extractor=mock_extractor,
        embed_fn=mock_embed_fn,
        openrouter_key="",
    )

    mock_stm = MagicMock()
    mock_stm.ingest = MagicMock(return_value=MagicMock(entities=0, chunks=0))

    mock_flush_result = MagicMock()
    mock_flush_result.nodes_written = 0
    mock_flush_result.edges_written = 0
    mock_flush_result.errors = []

    mock_flusher = MagicMock()
    mock_flusher.flush = AsyncMock(return_value=mock_flush_result)

    request = CorpusIngestRequest(documents=[])

    with (
        patch("graphknows.memory.stm.service.STMService", return_value=mock_stm),
        patch(
            "graphknows.memory.flush.SessionFlusher",
            return_value=mock_flusher,
        ),
    ):
        await workflow.run(request)

    # pg.connect() must have been called exactly once inside run()
    mock_pg.connect.assert_called_once()
