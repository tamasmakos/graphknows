"""Tests for wiring ADR-002 workflow via build_workflow factory."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from services.agent.graph.state import RawDoc, build_initial_investigative_state
from services.agent.graph.workflow import build_workflow


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading heavy models in unit tests."""
    with patch("services.agent.graph.nodes.investigate.SentenceTransformer") as mock_cls:
        mock_embedder = MagicMock()
        def fake_encode(texts, show_progress_bar=False):
            dim = 384
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.random.rand(n, dim).astype(np.float32)
            # Normalise so cosine-similarity stays bounded
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return arr / norms
        mock_embedder.encode = fake_encode
        mock_cls.return_value = mock_embedder
        yield


class TestBuildWorkflowWithoutDeps:
    """build_workflow with no external dependencies (mg_driver=None)."""

    def test_returns_compiled_graph(self) -> None:
        workflow = build_workflow(mg_driver=None)
        assert workflow is not None
        assert hasattr(workflow, "ainvoke")


@pytest.mark.asyncio
@patch("services.agent.graph.nodes.communicate.ChatOpenAI")
async def test_invoke_returns_briefing(mock_chat: MagicMock) -> None:
    """Invoking the graph with a simple topic returns a state with briefing set."""
    mock_instance = mock_chat.return_value
    mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Test briefing output."))

    session_id = "test-session-001"
    topic = "EU AI Act regulatory framework"
    initial_state = build_initial_investigative_state(session_id, topic)
    workflow = build_workflow(mg_driver=None)
    result = await workflow.ainvoke(initial_state)
    assert isinstance(result, dict)
    assert "briefing" in result
    assert result["briefing"] is not None
    assert len(result["briefing"]) > 0


@pytest.mark.asyncio
@patch("services.agent.graph.nodes.communicate.ChatOpenAI")
async def test_invoke_state_shape(mock_chat: MagicMock) -> None:
    """Invoking the graph returns full InvestigativeState keys."""
    mock_instance = mock_chat.return_value
    mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Briefing."))

    session_id = "test-session-002"
    initial_state = build_initial_investigative_state(session_id, "test topic")
    workflow = build_workflow()
    result = await workflow.ainvoke(initial_state)
    expected_keys = {
        "session_id", "topic", "plan", "queries", "raw_docs", "chunks",
        "entities", "relations", "iteration_count", "dup_ratio",
        "retrieval_hits", "briefing", "ws_events",
    }
    assert expected_keys.issubset(result.keys())
    assert result["session_id"] == session_id


class TestBuildWorkflowWithMocks:
    """build_workflow with mock mg_driver."""

    @pytest.fixture
    def mock_mg(self) -> MagicMock:
        mg = MagicMock()
        mg.session.return_value.__enter__.return_value = MagicMock()
        return mg

    @pytest.fixture
    def mock_search_client(self) -> MagicMock:
        sc = MagicMock()
        sc.fetch_documents = AsyncMock(return_value=[
            RawDoc(id="d1", title="Real Doc", text="Real content for testing.", url="http://ex.com/1", source="eurlex", date="2024-01-01"),
        ])
        return sc

    def test_returns_compiled_graph_with_deps(self, mock_mg: MagicMock) -> None:
        workflow = build_workflow(mg_driver=mock_mg)
        assert workflow is not None
        assert hasattr(workflow, "ainvoke")

    @pytest.mark.asyncio
    @patch("services.agent.graph.nodes.communicate.ChatOpenAI")
    async def test_invoke_with_mock_deps(self, mock_chat: MagicMock, mock_mg: MagicMock, mock_search_client: MagicMock) -> None:
        mock_instance = mock_chat.return_value
        mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Mock briefing."))

        session_id = "test-session-003"
        initial_state = build_initial_investigative_state(session_id, "EU digital policy")
        workflow = build_workflow(mg_driver=mock_mg, search_client=mock_search_client)
        result = await workflow.ainvoke(initial_state)
        assert result["briefing"] is not None
        assert len(result.get("ws_events", [])) > 0
        assert mock_search_client.fetch_documents.await_count >= 1
