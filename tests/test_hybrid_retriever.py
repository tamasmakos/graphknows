"""Unit tests for GraphRAGClient — the unified HTTP client for graphrag service."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.tools.graph.client import GraphRAGClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


class TestGraphRAGClientRetrieve:
    """Tests for the retrieve() method."""

    @pytest.mark.asyncio
    async def test_retrieve_posts_to_retrieve_endpoint(self):
        expected = {"chunks": [{"id": "c1", "text": "hello"}], "entities": [], "scores": []}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            result = await client.retrieve("EU regulation", "session-1", top_k=5)

            mock_client.post.assert_awaited_once_with(
                "/retrieve",
                json={"query": "EU regulation", "session_id": "session-1", "top_k": 5},
            )
            assert result == expected

    @pytest.mark.asyncio
    async def test_retrieve_default_top_k(self):
        expected = {"chunks": [], "entities": [], "scores": []}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            await client.retrieve("query", "sid")

            call_json = mock_client.post.call_args[1]["json"]
            assert call_json["top_k"] == 10  # default


# ---------------------------------------------------------------------------
# get_path()
# ---------------------------------------------------------------------------


class TestGraphRAGClientGetPath:
    """Tests for the get_path() method."""

    @pytest.mark.asyncio
    async def test_get_path_posts_correct_payload(self):
        expected = {"path": ["EntityA", "EntityB"], "length": 1}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            result = await client.get_path("EntityA", "EntityB", max_hops=2)

            mock_client.post.assert_awaited_once_with(
                "/path",
                json={"entity_a": "EntityA", "entity_b": "EntityB", "max_hops": 2},
            )
            assert result["path"] == ["EntityA", "EntityB"]


# ---------------------------------------------------------------------------
# get_chunks()
# ---------------------------------------------------------------------------


class TestGraphRAGClientGetChunks:
    """Tests for the get_chunks() method."""

    @pytest.mark.asyncio
    async def test_get_chunks_posts_correct_payload(self):
        expected = {"chunks": ["text about entity"]}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            result = await client.get_chunks("EuropeanParliament", "sid-42", top_k=3)

            mock_client.post.assert_awaited_once_with(
                "/chunks",
                json={"entity": "EuropeanParliament", "session_id": "sid-42", "top_k": 3},
            )
            assert result == expected


# ---------------------------------------------------------------------------
# get_community()
# ---------------------------------------------------------------------------


class TestGraphRAGClientGetCommunity:
    """Tests for the get_community() method."""

    @pytest.mark.asyncio
    async def test_get_community_posts_correct_payload(self):
        expected = {"nodes": [{"id": "EntityA"}], "edges": []}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            result = await client.get_community("EntityA", depth=2)

            mock_client.post.assert_awaited_once_with(
                "/community",
                json={"entity": "EntityA", "depth": 2},
            )
            assert result["nodes"] == [{"id": "EntityA"}]


# ---------------------------------------------------------------------------
# ingest_chunks()
# ---------------------------------------------------------------------------


class TestGraphRAGClientIngestChunks:
    """Tests for the ingest_chunks() method."""

    @pytest.mark.asyncio
    async def test_ingest_posts_chunks(self):
        chunks = [{"id": "c1", "text": "hello world"}]
        expected = {"ingested": 1}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(expected)

            client = GraphRAGClient("http://graphrag:8010")
            result = await client.ingest_chunks(chunks, "session-xyz")

            mock_client.post.assert_awaited_once_with(
                "/ingest",
                json={"chunks": chunks, "session_id": "session-xyz"},
            )
            assert result["ingested"] == 1


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------


class TestGraphRAGClientContextManager:
    """Tests for async context manager behaviour."""

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self):
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.aclose = AsyncMock()
            mock_cls.return_value = mock_instance

            grc = GraphRAGClient("http://graphrag:8010")
            async with grc:
                assert grc._client is mock_instance

            mock_instance.aclose.assert_awaited_once()
            assert grc._client is None

    def test_base_url_trailing_slash_stripped(self):
        grc = GraphRAGClient("http://graphrag:8010/")
        assert grc._base_url == "http://graphrag:8010"
