"""Unit tests for GraphknowsAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.adapter import GraphknowsAdapter
from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    MemoryClass,
    ParticipantRecord,
    PromoteOptions,
    RecallRequest,
    ToolCallRecord,
    TurnRecord,
)
from graphknows.retrieval.recall_stm import RecallResult


def _make_adapter():
    mg = MagicMock()
    pg = MagicMock()
    neo = MagicMock()
    extractor = MagicMock()
    embed_fn = MagicMock(return_value=[0.1] * 10)
    return GraphknowsAdapter(mg, pg, neo, extractor, embed_fn)


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_open_session_sets_active_session_id(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    adapter.open_session("s1")
    assert adapter._active_session_id == "s1"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_start_dialog_delegates_to_dialog_writer(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    participants = [
        ParticipantRecord(
            participant_id="p1",
            role="user",
            display_name="Alice",
        )
    ]
    adapter.start_dialog("s1", "d1", participants)

    assert adapter._active_dialog_id == "d1"
    mock_dialog_writer.return_value.start_dialog.assert_called_once()
    args, kwargs = mock_dialog_writer.return_value.start_dialog.call_args
    assert args[0] == "s1"
    assert args[1] == "d1"
    mapped_participants = args[2]
    assert len(mapped_participants) == 1
    assert mapped_participants[0].id == "p1"
    assert mapped_participants[0].role == "user"
    assert mapped_participants[0].display_name == "Alice"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_write_turn_delegates_to_dialog_writer(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    adapter._active_dialog_id = "d1"
    turn = TurnRecord(
        turn_id="t1",
        session_id="s1",
        role="user",
        content="Hello",
        tool_calls=[
            ToolCallRecord(
                tool_id="tc1",
                tool_name="search",
                input_summary="query",
                output_summary="result",
            )
        ],
    )
    adapter.write_turn(turn)

    mock_dialog_writer.return_value.write_turn.assert_called_once()
    args, kwargs = mock_dialog_writer.return_value.write_turn.call_args
    assert args[0] == "s1"
    assert args[1] == "d1"
    mapped_turn = args[2]
    assert mapped_turn.role == "user"
    assert mapped_turn.text == "Hello"
    assert len(mapped_turn.tool_calls) == 1
    assert mapped_turn.tool_calls[0].name == "search"
    assert mapped_turn.tool_calls[0].result_summary == "result"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_ingest_artifact_delegates_to_ingestor(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    artifact = ArtifactRecord(
        artifact_id="a1",
        path="/tmp/test.py",
        content="print('hello')",
        mime_type="text/x-python",
        memory_class=MemoryClass.raw_evidence,
    )
    adapter.ingest_artifact(artifact, "s1")

    mock_ingestor.return_value.ingest_text.assert_called_once()
    args, kwargs = mock_ingestor.return_value.ingest_text.call_args
    assert args[0] == "print('hello')"
    assert kwargs.get("session_id") == "s1"
    assert kwargs.get("title") == "/tmp/test.py"
    assert kwargs.get("source_hint") == "text/x-python"
    metadata = kwargs.get("metadata", {})
    assert metadata.get("memory_class") == "raw_evidence"
    assert metadata.get("artifact_id") == "a1"


@patch("graphknows.retrieval.recall_stm.recall_short_term")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_recall_returns_hits_as_dicts(
    mock_dialog_writer,
    mock_ingestor,
    mock_recall,
):
    adapter = _make_adapter()
    mock_recall.return_value = [
        RecallResult(
            text="hello",
            source_type="chunk",
            scope="short_term",
            score=0.9,
        )
    ]
    request = RecallRequest(query="test", session_id="s1", top_k=5)
    result = adapter.recall(request)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["text"] == "hello"
    assert result[0]["source_type"] == "chunk"
    assert result[0]["scope"] == "short_term"


@patch("graphknows.retrieval.recall_stm.recall_short_term")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_recall_filters_raw_evidence_when_include_raw_false(
    mock_dialog_writer,
    mock_ingestor,
    mock_recall,
):
    adapter = _make_adapter()
    mock_recall.return_value = [
        RecallResult(
            text="raw",
            source_type="chunk",
            scope="short_term",
            score=0.9,
            provenance={"memory_class": "raw_evidence"},
        ),
        RecallResult(
            text="summary",
            source_type="chunk",
            scope="short_term",
            score=0.8,
            provenance={"memory_class": "summary"},
        ),
    ]
    request = RecallRequest(
        query="test",
        session_id="s1",
        top_k=5,
        include_raw=False,
    )
    result = adapter.recall(request)

    assert len(result) == 1
    assert result[0]["text"] == "summary"


@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
def test_capabilities_returns_all_true(
    mock_dialog_writer,
    mock_ingestor,
):
    adapter = _make_adapter()
    caps = adapter.capabilities()

    assert isinstance(caps, AdapterCapabilities)
    assert caps.supports_stm is True
    assert caps.supports_ltm is True
    assert caps.supports_dialog_graph is True
    assert caps.supports_artifact_ingest is True
    assert caps.supports_recall is True
    assert caps.supports_promote is True


@patch("graphknows.memory.adapter.SessionFlusher")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
@pytest.mark.asyncio
async def test_promote_calls_session_flusher(
    mock_dialog_writer,
    mock_ingestor,
    mock_flusher,
):
    adapter = _make_adapter()
    mock_flusher.return_value.flush = AsyncMock(
        return_value=MagicMock(
            nodes_written=10,
            edges_written=5,
            chunks_promoted=3,
            turns_promoted=2,
            topics_created=1,
            merged_entities=0,
            elapsed_s=1.5,
            errors=[],
        )
    )
    options = PromoteOptions(
        session_id="s1",
        run_gds=True,
        run_topics=True,
    )
    result = await adapter.promote(options)

    mock_flusher.return_value.flush.assert_awaited_once()
    args, kwargs = mock_flusher.return_value.flush.call_args
    assert args[0] == "s1"
    assert result["session_id"] == "s1"
    assert result["run_gds"] is True
    assert result["run_topics"] is True
    assert result["nodes_written"] == 10
    assert result["edges_written"] == 5
    assert result["chunks_promoted"] == 3
    assert result["turns_promoted"] == 2
    assert result["topics_created"] == 1
    assert result["merged_entities"] == 0
    assert result["elapsed_s"] == 1.5
    assert result["errors"] == []
