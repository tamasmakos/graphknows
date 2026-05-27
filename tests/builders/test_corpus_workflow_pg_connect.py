"""Test build_corpus_workflow connects the pg store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def test_build_corpus_workflow_connects_pg() -> None:
    """build_corpus_workflow must call pg.connect() before returning."""
    mock_pg = MagicMock()
    mock_pg.connect = AsyncMock(return_value=None)

    with (
        patch("graphknows.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.builders.stores.build_pgvector_store", return_value=mock_pg),
        patch("graphknows.builders.stores.build_neo4j_store", return_value=MagicMock()),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor", MagicMock()),
        patch("graphknows.memory.ingestor._run", side_effect=lambda coro: None),
        patch(
            "graphknows.memory.workflows.corpus_ingest.CorpusIngestWorkflow",
            return_value=MagicMock(),
        ),
        patch("graphknows.retrieval.stores.embedder.embed_one", MagicMock()),
    ):
        from graphknows.builders.stores import build_corpus_workflow
        from graphknows.config import GraphKnowsSettings

        settings = MagicMock(spec=GraphKnowsSettings)
        settings.openrouter_api_key = None

        build_corpus_workflow(settings)

        # pg.connect() must have been called (via _run)
        mock_pg.connect.assert_called_once()
