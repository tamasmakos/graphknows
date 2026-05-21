"""Unit tests for HybridRAGClient — the unified HTTP client for hybridrag service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.tools.graph.client import HybridRAGClient


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestHybridRAGClientRetrieve:
    """Tests for the retrieve() method."""

    @pytest.mark.asyncio
    async def test_retrieve_posts_to_retrieve_endpoint_and_returns_str(self):
        xml_result = "<retrieval_context></retrieval_context>"
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(xml_result)

            client = HybridRAGClient("http://hybridrag:8020")
            result = await client.retrieve(
                "EU regulation", "session-1", k_entity=5, k_semantic=10, k_final=5
            )

            mock_client.post.assert_awaited_once_with(
                "/retrieve",
                json={
                    "query": "EU regulation",
                    "session_id": "session-1",
                    "k_entity": 5,
                    "k_semantic": 10,
                    "k_final": 5,
                },
            )
            assert result == xml_result

    @pytest.mark.asyncio
    async def test_retrieve_default_params(self):
        xml_result = "<retrieval_context></retrieval_context>"
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.post.return_value = _mock_response(xml_result)

            client = HybridRAGClient("http://hybridrag:8020")
            await client.retrieve("query", "sid")

            call_json = mock_client.post.call_args[1]["json"]
            assert call_json["k_entity"] == 10
            assert call_json["k_semantic"] == 20
            assert call_json["k_final"] == 10


class TestHybridRAGClientCommunity:
    """Tests for the community() method."""

    @pytest.mark.asyncio
    async def test_community_calls_get_endpoint(self):
        expected = {"id": "c1", "label": "ORG", "summary": "Financial sector entities."}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.get.return_value = _mock_response(expected)

            client = HybridRAGClient("http://hybridrag:8020")
            result = await client.community("EuropeanParliament")

            mock_client.get.assert_awaited_once_with("/community/EuropeanParliament")
            assert result["label"] == "ORG"


class TestHybridRAGClientHealth:
    """Tests for the health() method."""

    @pytest.mark.asyncio
    async def test_health_calls_get_health(self):
        expected = {"status": "ok"}
        with patch("services.agent.tools.graph.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_cls.return_value.__aexit__.return_value = False
            mock_client.get.return_value = _mock_response(expected)

            client = HybridRAGClient("http://hybridrag:8020")
            result = await client.health()

            mock_client.get.assert_awaited_once_with("/health")
            assert result["status"] == "ok"

    def test_base_url_trailing_slash_stripped(self):
        client = HybridRAGClient("http://hybridrag:8020/")
        assert client._base_url == "http://hybridrag:8020"
