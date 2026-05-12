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


def _mock_chatgroq():
    """Return a mock ChatGroq instance with async ainvoke."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "# Mock Briefing\n\n**EPRS Intelligence Brief**"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


@pytest.fixture(autouse=True)
def _patch_chatgroq():
    """Auto-patch ChatGroq so tests don't need GROQ_API_KEY."""
    with patch("services.agent.graph.nodes.communicate.ChatGroq") as mock_cls:
        mock_cls.return_value = _mock_chatgroq()
        yield


class TestCommunicateNodeNoRetriever:
    """Tests for communicate_node without a retriever (fallback path)."""

    @pytest.mark.asyncio
    async def test_fallback_returns_first_8_chunks(self):
        chunks = [Chunk(id=f"c{i}", doc_id=f"d{i}", text=f"text{i}") for i in range(12)]
        state = _make_state(chunks)
        result = await communicate_node(state, retriever=None)
        assert len(result["retrieval_hits"]) == 8
        assert result["retrieval_hits"] == chunks[:8]

    @pytest.mark.asyncio
    async def test_fallback_with_less_than_8_chunks(self):
        chunks = [Chunk(id=f"c{i}", doc_id=f"d{i}", text=f"text{i}") for i in range(3)]
        state = _make_state(chunks)
        result = await communicate_node(state, retriever=None)
        assert len(result["retrieval_hits"]) == 3

    @pytest.mark.asyncio
    async def test_fallback_with_no_chunks(self):
        state = _make_state([])
        result = await communicate_node(state, retriever=None)
        assert result["retrieval_hits"] == []

    @pytest.mark.asyncio
    async def test_ws_events_contain_retrieval_done(self):
        state = _make_state([Chunk(id="c1", doc_id="d1", text="hello")])
        result = await communicate_node(state, retriever=None)
        types = [e["event_type"] for e in result["ws_events"]]
        assert "retrieval_done" in types

    @pytest.mark.asyncio
    async def test_ws_events_contain_brief_done(self):
        state = _make_state([Chunk(id="c1", doc_id="d1", text="hello")])
        result = await communicate_node(state, retriever=None)
        types = [e["event_type"] for e in result["ws_events"]]
        assert "brief_done" in types


class TestCommunicateNodeWithRetriever:
    """Tests for communicate_node with a mock retriever."""

    @pytest.fixture
    def mock_retriever(self):
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(
            return_value=[
                Chunk(id="r1", doc_id="rd1", text="retrieved text 1"),
                Chunk(id="r2", doc_id="rd2", text="retrieved text 2"),
            ]
        )
        return retriever

    @pytest.mark.asyncio
    async def test_uses_retriever_when_provided(self, mock_retriever):
        state = _make_state([])
        result = await communicate_node(state, retriever=mock_retriever)
        mock_retriever.retrieve.assert_awaited_once()
        assert len(result["retrieval_hits"]) == 2

    @pytest.mark.asyncio
    async def test_retriever_called_with_topic(self, mock_retriever):
        state = _make_state([])
        await communicate_node(state, retriever=mock_retriever)
        mock_retriever.retrieve.assert_awaited_once_with(
            "EU AI regulation", k_dense=20, k_final=8
        )

    @pytest.mark.asyncio
    async def test_preserves_existing_ws_events(self, mock_retriever):
        state = _make_state([])
        state["ws_events"] = [{"event_type": "plan_ready", "session_id": "test-sid", "payload": {}, "ts": 0.0}]
        result = await communicate_node(state, retriever=mock_retriever)
        assert len(result["ws_events"]) == 3  # plan_ready + retrieval_done + brief_done


class TestMakeCommunicateNode:
    """Tests for the factory function make_communicate_node."""

    @pytest.mark.asyncio
    async def test_factory_returns_callable(self):
        node = make_communicate_node(retriever=None)
        assert callable(node)

    @pytest.mark.asyncio
    async def test_factory_binds_retriever(self):
        retriever = AsyncMock()
        retriever.retrieve = AsyncMock(return_value=[])
        node = make_communicate_node(retriever=retriever)
        state = _make_state([])
        await node(state)
        retriever.retrieve.assert_awaited_once()
