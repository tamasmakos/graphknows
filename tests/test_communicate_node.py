"""Unit tests for communicate_node -- briefing generation."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.graph.nodes.communicate import (
    communicate_node,
    make_communicate_node,
)
from services.agent.graph.state import Chunk, build_initial_investigative_state


def _make_state(chunks: list | None = None) -> dict:
    state = dict(build_initial_investigative_state("test-sid", "EU AI regulation"))
    if chunks is not None:
        state["chunks"] = chunks
    state["ws_events"] = []
    return state


def _mock_chatopenai():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# Mock Briefing\n\n**EPRS Intelligence Brief**"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


_SAMPLE_XML = (
    '<retrieval_context session_id="test-sid" query="EU AI regulation" total_chunks="2">'
    '<graph_matches count="0"></graph_matches>'
    '<fused_results count="2">'
    '<chunk id="c1" doc_id="d1" rrf_score="0.016" sources="semantic" rank="1">'
    '<text>Retrieved chunk 1</text></chunk>'
    '<chunk id="c2" doc_id="d2" rrf_score="0.016" sources="semantic" rank="2">'
    '<text>Retrieved chunk 2</text></chunk>'
    '</fused_results>'
    '</retrieval_context>'
)


@pytest.fixture(autouse=True)
def _patch_chatopenai():
    with patch("services.agent.graph.nodes.communicate.ChatOpenAI") as mock_cls:
        mock_cls.return_value = _mock_chatopenai()
        yield


class TestCommunicateNodeHybridRAG:
    """Tests for communicate_node calling hybridrag HTTP service."""

    @pytest.mark.asyncio
    async def test_calls_hybridrag_and_returns_briefing(self):
        state = _make_state([])
        with patch("services.agent.tools.graph.client.HybridRAGClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.retrieve = AsyncMock(return_value=_SAMPLE_XML)
            mock_cls.return_value = mock_instance
            result = await communicate_node(state)
        assert "briefing" in result
        assert result["briefing"]

    @pytest.mark.asyncio
    async def test_fallback_to_state_chunks_on_error(self):
        chunks = [Chunk(id=f"c{i}", doc_id=f"d{i}", text=f"text{i}") for i in range(4)]
        state = _make_state(chunks)
        with patch(
            "services.agent.tools.graph.client.HybridRAGClient",
            side_effect=Exception("service down"),
        ):
            result = await communicate_node(state)
        assert "briefing" in result

    @pytest.mark.asyncio
    async def test_ws_events_contain_retrieval_done(self):
        state = _make_state([])
        with patch("services.agent.tools.graph.client.HybridRAGClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.retrieve = AsyncMock(return_value=_SAMPLE_XML)
            mock_cls.return_value = mock_instance
            result = await communicate_node(state)
        types = [e["event_type"] for e in result["ws_events"]]
        assert "retrieval_done" in types

    @pytest.mark.asyncio
    async def test_ws_events_contain_brief_done(self):
        state = _make_state([])
        with patch("services.agent.tools.graph.client.HybridRAGClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.retrieve = AsyncMock(return_value=_SAMPLE_XML)
            mock_cls.return_value = mock_instance
            result = await communicate_node(state)
        types = [e["event_type"] for e in result["ws_events"]]
        assert "brief_done" in types

    @pytest.mark.asyncio
    async def test_preserves_existing_ws_events(self):
        state = _make_state([])
        state["ws_events"] = [
            {"event_type": "plan_ready", "session_id": "test-sid", "payload": {}, "ts": 0.0}
        ]
        with patch("services.agent.tools.graph.client.HybridRAGClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.retrieve = AsyncMock(return_value=_SAMPLE_XML)
            mock_cls.return_value = mock_instance
            result = await communicate_node(state)
        assert len(result["ws_events"]) == 3  # plan_ready + retrieval_done + brief_done


class TestMakeCommunicateNode:
    """Tests for the make_communicate_node factory."""

    @pytest.mark.asyncio
    async def test_factory_returns_callable(self):
        node = make_communicate_node()
        assert callable(node)

    @pytest.mark.asyncio
    async def test_factory_node_runs(self):
        node = make_communicate_node()
        state = _make_state([])
        with patch("services.agent.tools.graph.client.HybridRAGClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.retrieve = AsyncMock(return_value=_SAMPLE_XML)
            mock_cls.return_value = mock_instance
            result = await node(state)
        assert "briefing" in result
