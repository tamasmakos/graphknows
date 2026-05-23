"""Conformance tests for GraphknowsAdapter against the synthetic host fixture."""
from __future__ import annotations

import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

from graphknows.memory.adapter import GraphknowsAdapter
from graphknows.memory.adapter_types import AdapterCapabilities, RecallRequest
from graphknows.retrieval.recall_stm import RecallResult

from tests.fixtures.synthetic_host_session import (
    ARTIFACTS,
    DIALOG_ID,
    PARTICIPANTS,
    PROMOTE_OPTIONS,
    RECALL_REQUEST,
    SESSION_ID,
    TURNS,
)


def _make_adapter():
    """Return a GraphknowsAdapter with all stores mocked."""
    return GraphknowsAdapter(
        mg_store=MagicMock(),
        pg_store=MagicMock(),
        neo_store=MagicMock(),
        extractor=MagicMock(),
        embed_fn=None,
    )


class TestAdapterLifecycle:
    """Conformance tests for full adapter lifecycle sequences."""

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_full_session_write_sequence(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        assert adapter._active_session_id == SESSION_ID

        adapter.start_dialog(SESSION_ID, DIALOG_ID, PARTICIPANTS)
        mock_dw_cls.return_value.start_dialog.assert_called_once()

        for turn in TURNS:
            adapter.write_turn(turn)
        assert mock_dw_cls.return_value.write_turn.call_count == len(TURNS)

        for artifact in ARTIFACTS:
            adapter.ingest_artifact(artifact, SESSION_ID)
        assert mock_ing_cls.return_value.ingest_text.call_count == len(ARTIFACTS)

    @patch("graphknows.retrieval.recall_stm.recall_short_term")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_recall_after_write(self, mock_dw_cls, mock_ing_cls, mock_recall):
        mock_recall.return_value = [
            RecallResult(
                text="EU AI Act fact",
                source_type="chunk",
                scope="short_term",
                score=0.9,
                metadata={"memory_class": "fact"},
            )
        ]
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.start_dialog(SESSION_ID, DIALOG_ID, PARTICIPANTS)
        for turn in TURNS:
            adapter.write_turn(turn)

        result = adapter.recall(RECALL_REQUEST)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "EU AI Act fact"
        assert "source_type" in result[0]

    @patch("graphknows.memory.adapter.SessionFlusher")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    @pytest.mark.asyncio
    async def test_promote_after_write(self, mock_dw_cls, mock_ing_cls, mock_flusher_cls):
        mock_flush_result = MagicMock()
        mock_flush_result.__dict__ = {"session_id": SESSION_ID}
        mock_flusher_cls.return_value.flush = AsyncMock(return_value=mock_flush_result)

        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.write_turn(TURNS[0])

        result = await adapter.promote(PROMOTE_OPTIONS)
        mock_flusher_cls.return_value.flush.assert_called_once()
        call_session_id = mock_flusher_cls.return_value.flush.call_args[0][0]
        assert call_session_id == SESSION_ID
        assert isinstance(result, dict)

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_capabilities_contract(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        caps = adapter.capabilities()
        assert isinstance(caps, AdapterCapabilities)
        assert caps.supports_stm is True
        assert caps.supports_ltm is True
        assert caps.supports_dialog_graph is True
        assert caps.supports_artifact_ingest is True
        assert caps.supports_recall is True
        assert caps.supports_promote is True

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_turn_with_tool_calls_maps_correctly(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.start_dialog(SESSION_ID, DIALOG_ID, PARTICIPANTS)
        turn_with_tool = TURNS[1]  # has one tool_call
        adapter.write_turn(turn_with_tool)

        call_args = mock_dw_cls.return_value.write_turn.call_args
        # positional args: (session_id, dialog_id, Turn)
        written_turn = call_args[0][2]
        assert len(written_turn.tool_calls) == 1
        assert written_turn.tool_calls[0].name == "web_search"

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_artifact_metadata_includes_memory_class(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.ingest_artifact(ARTIFACTS[0], SESSION_ID)

        call_kwargs = mock_ing_cls.return_value.ingest_text.call_args[1]
        assert call_kwargs["metadata"]["memory_class"] == "raw_evidence"

    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_open_session_idempotent(self, mock_dw_cls, mock_ing_cls):
        adapter = _make_adapter()
        adapter.open_session(SESSION_ID)
        adapter.open_session(SESSION_ID)  # second call — no exception
        assert adapter._active_session_id == SESSION_ID

    @patch("graphknows.retrieval.recall_stm.recall_short_term")
    @patch("graphknows.memory.adapter.MemoryIngestor")
    @patch("graphknows.memory.adapter.DialogWriter")
    def test_recall_include_raw_false_filters(self, mock_dw_cls, mock_ing_cls, mock_recall):
        mock_recall.return_value = [
            RecallResult(
                text="raw hit",
                source_type="chunk",
                scope="short_term",
                score=0.9,
                provenance={"memory_class": "raw_evidence"},
            ),
            RecallResult(
                text="fact hit",
                source_type="chunk",
                scope="short_term",
                score=0.8,
                provenance={"memory_class": "fact"},
            ),
        ]
        adapter = _make_adapter()
        req = RecallRequest(query="test", session_id=SESSION_ID, top_k=5, include_raw=False)
        result = adapter.recall(req)
        texts = [r["text"] for r in result]
        assert "raw hit" not in texts
        assert "fact hit" in texts
