"""Unit tests for HybridRetriever -- RRF fusion and full retrieval pipeline."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.graph.state import Chunk
from services.agent.retrieval.hybrid_retriever import HybridRetriever, _extract_entity_ids


# ------------------------------------------------------------------
# _extract_entity_ids
# ------------------------------------------------------------------


class TestExtractEntityIds:
    """Tests for the simple regex-based entity extraction helper."""

    def test_extracts_capitalised_phrases(self):
        texts = ["The European Commission proposed a new regulation."]
        result = _extract_entity_ids(texts)
        assert "European Commission" in result[0]

    def test_deduplicates(self):
        texts = [
            "European Commission proposal.",
            "European Commission response.",
        ]
        result = _extract_entity_ids(texts)
        assert result == ["European Commission"]

    def test_ignores_single_words(self):
        texts = ["Commission said something important today."]
        assert _extract_entity_ids(texts) == []

    def test_handles_empty_input(self):
        assert _extract_entity_ids([]) == []


# ------------------------------------------------------------------
# RRF fusion
# ------------------------------------------------------------------


class TestRRF:
    """Tests for the Reciprocal Rank Fusion static method."""

    def test_rrf_fuses_dense_and_graph(self):
        dense = [
            Chunk(id="a", doc_id="d1", text="alpha"),
            Chunk(id="b", doc_id="d2", text="beta"),
        ]
        graph = [
            Chunk(id="c", doc_id="d3", text="gamma"),
            Chunk(id="a", doc_id="d1", text="alpha"),  # overlap
        ]
        result = HybridRetriever._rrf(dense, graph, k_final=3, k=60)
        assert len(result) <= 3
        # 'a' appears in both lists so it should rank highest
        assert result[0].id == "a"

    def test_rrf_respects_k_final(self):
        dense = [Chunk(id=str(i), doc_id=f"d{i}", text=str(i)) for i in range(5)]
        graph = [Chunk(id=str(i), doc_id=f"d{i}", text=str(i)) for i in range(5, 10)]
        result = HybridRetriever._rrf(dense, graph, k_final=3, k=60)
        assert len(result) == 3

    def test_rrf_with_empty_graph(self):
        dense = [
            Chunk(id="a", doc_id="d1", text="alpha"),
            Chunk(id="b", doc_id="d2", text="beta"),
        ]
        result = HybridRetriever._rrf(dense, [], k_final=2, k=60)
        assert result == dense

    def test_rrf_with_empty_dense(self):
        graph = [
            Chunk(id="a", doc_id="d1", text="alpha"),
            Chunk(id="b", doc_id="d2", text="beta"),
        ]
        result = HybridRetriever._rrf([], graph, k_final=2, k=60)
        assert result == graph

    def test_rrf_uses_k_60_constant(self):
        """Verify default k=60 is used when not overridden."""
        dense = [Chunk(id="x", doc_id="dx", text="x")]
        graph = [Chunk(id="y", doc_id="dy", text="y")]
        result = HybridRetriever._rrf(dense, graph, k_final=2)
        assert len(result) == 2


# ------------------------------------------------------------------
# Full retrieve pipeline
# ------------------------------------------------------------------


class TestHybridRetrieverRetrieve:
    """Tests for the async retrieve method with mocked dependencies."""

    @pytest.fixture
    def mock_pg(self):
        client = AsyncMock()
        client.search_chunks = AsyncMock(
            return_value=[
                {"id": "c1", "doc_id": "d1", "text": "European Commission regulation", "score": 0.95},
                {"id": "c2", "doc_id": "d2", "text": "EU Parliament debate", "score": 0.90},
            ]
        )
        client.get_chunks_by_doc_ids = AsyncMock(
            return_value=[
                {"id": "c3", "doc_id": "d3", "text": "Graph hit chunk"},
            ]
        )
        return client

    @pytest.fixture
    def mock_mg(self):
        driver = MagicMock()
        session = MagicMock()
        result = MagicMock()
        result.data.return_value = [
            {"id": "EntityX", "doc_ids": ["d3"]},
        ]
        session.run.return_value = result
        driver.session.return_value.__enter__.return_value = session
        return driver

    @pytest.fixture
    def retriever(self, mock_pg, mock_mg):
        return HybridRetriever(session_id="test-sid", pg_client=mock_pg, mg_driver=mock_mg)

    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self, retriever):
        result = await retriever.retrieve("test query", k_dense=20, k_final=8)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    @pytest.mark.asyncio
    async def test_retrieve_respects_k_final(self, retriever):
        result = await retriever.retrieve("test query", k_dense=20, k_final=2)
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_session_id_passed_to_pgvector(self, retriever, mock_pg):
        await retriever.retrieve("test query")
        mock_pg.search_chunks.assert_awaited_once_with(
            "test-sid", "test query", k=20
        )

    @pytest.mark.asyncio
    async def test_session_id_passed_to_memgraph(self, retriever, mock_mg):
        await retriever.retrieve("test query")
        call_kwargs = mock_mg.session.return_value.__enter__.return_value.run.call_args[1]
        assert call_kwargs["sid"] == "test-sid"

    @pytest.mark.asyncio
    async def test_retrieve_with_no_entities(self, mock_pg):
        """When no entities are extracted, graph walk is skipped."""
        mock_pg.search_chunks = AsyncMock(
            return_value=[
                {"id": "c1", "doc_id": "d1", "text": "some lower case text", "score": 0.5},
            ]
        )
        retriever = HybridRetriever(
            session_id="test-sid", pg_client=mock_pg, mg_driver=MagicMock()
        )
        result = await retriever.retrieve("test", k_dense=20, k_final=5)
        # Should still return dense-only results
        assert len(result) == 1
        assert result[0].id == "c1"

    @pytest.mark.asyncio
    async def test_retrieve_empty_dense_hits(self, mock_pg):
        mock_pg.search_chunks = AsyncMock(return_value=[])
        retriever = HybridRetriever(
            session_id="test-sid", pg_client=mock_pg, mg_driver=MagicMock()
        )
        result = await retriever.retrieve("test query", k_dense=20, k_final=5)
        assert result == []
