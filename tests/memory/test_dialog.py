"""Unit tests for DialogWriter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from graphknows.memory.dialog import DialogWriter, Participant, ToolCall, Turn
from graphknows.memory.extractor import ExtractionResult


def _make_writer(extractor=None, embed_fn=None):
    mg = MagicMock()
    pg = MagicMock()
    # All Memgraph and pgvector store methods are async — mock them as coroutines
    for attr in [
        "write_dialog_node",
        "write_entity_node",
        "write_turn_node",
        "write_turn_sequence",
        "write_turn_entity_edge",
        "write_tool_call_node",
        "write_chunk_node",
        "write_chunk_entity_edge",
        "write_session_metadata",
    ]:
        setattr(mg, attr, AsyncMock(return_value=None))
    pg.write_stm_turn = AsyncMock(return_value=None)
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    writer = DialogWriter(mg, pg, extractor=extractor, embed_fn=embed)
    return writer, mg, pg, embed


def _turn(index: int = 0, role: str = "user", text: str = "Hello") -> Turn:
    return Turn(id=f"t{index}", index=index, role=role, text=text)


def test_start_dialog_calls_write_dialog_node():
    writer, mg, pg, _ = _make_writer()
    writer.start_dialog("s1", "d1", [Participant("u1", "user")])
    mg.write_dialog_node.assert_called_once()
    # participants now written as entity nodes (write_participant_node removed)
    mg.write_entity_node.assert_called_once()


def test_write_turn_calls_write_turn_node():
    writer, mg, pg, _ = _make_writer()
    writer.write_turn("s1", "d1", _turn(0, "user", "Hi"))
    mg.write_turn_node.assert_called_once()
    _, kwargs = mg.write_turn_node.call_args
    assert kwargs.get("session_id") == "s1"


def test_write_turn_calls_next_edge_on_second_turn():
    writer, mg, pg, _ = _make_writer()
    writer.write_turn("s1", "d1", _turn(0))
    writer.write_turn("s1", "d1", _turn(1))
    # write_turn_sequence replaced write_turn_next_edge
    mg.write_turn_sequence.assert_called_once_with(prev_turn_id="t0", next_turn_id="t1")


def test_write_turn_no_next_edge_on_first_turn():
    writer, mg, pg, _ = _make_writer()
    writer.write_turn("s1", "d1", _turn(0))
    mg.write_turn_sequence.assert_not_called()


def test_write_turn_calls_extractor_when_set():
    ext = MagicMock()
    ext.extract = MagicMock(
        return_value=ExtractionResult(entities=[{"name": "EU", "type": "ORG", "score": 0.9}])
    )
    writer, mg, pg, _ = _make_writer(extractor=ext)
    writer.write_turn("s1", "d1", _turn(0, text="The EU signed a deal."))
    ext.extract.assert_called_once()


def test_write_turn_calls_write_entity_for_each_entity():
    ext = MagicMock()
    ext.extract = MagicMock(
        return_value=ExtractionResult(
            entities=[
                {"name": "EU", "type": "ORG", "score": 0.9},
                {"name": "France", "type": "LOC", "score": 0.8},
            ]
        )
    )
    writer, mg, pg, _ = _make_writer(extractor=ext)
    result = writer.write_turn("s1", "d1", _turn(0, text="EU and France"))
    assert result.entity_count == 2
    assert mg.write_entity_node.call_count == 2


def test_write_turn_calls_write_tool_call_node():
    writer, mg, pg, _ = _make_writer()
    turn = Turn(
        id="t0",
        index=0,
        role="agent",
        text="Using search tool.",
        tool_calls=[ToolCall(name="search_eu", latency_ms=120.0)],
    )
    writer.write_turn("s1", "d1", turn)
    mg.write_tool_call_node.assert_called_once()


def test_write_turn_calls_write_stm_turn():
    writer, mg, pg, _ = _make_writer()
    writer.write_turn("s1", "d1", _turn(0))
    pg.write_stm_turn.assert_called_once()


def test_extractor_failure_appended_to_errors():
    ext = MagicMock()
    ext.extract = MagicMock(side_effect=RuntimeError("NER crash"))
    writer, mg, pg, _ = _make_writer(extractor=ext)
    result = writer.write_turn("s1", "d1", _turn(0, text="Some text"))
    assert any("entity_extract" in e for e in result.errors)


def test_pg_failure_appended_to_errors():
    writer, mg, pg, _ = _make_writer()
    pg.write_stm_turn = AsyncMock(side_effect=RuntimeError("DB down"))
    result = writer.write_turn("s1", "d1", _turn(0))
    assert any("pg_stm_turn" in e for e in result.errors)


def test_dialog_result_entity_count_matches_extracted():
    ext = MagicMock()
    ext.extract = MagicMock(
        return_value=ExtractionResult(entities=[{"name": "Berlin", "type": "LOC", "score": 0.95}])
    )
    writer, mg, pg, _ = _make_writer(extractor=ext)
    result = writer.write_turn("s1", "d1", _turn(0, text="Berlin is a city."))
    assert result.entity_count == 1


def test_prev_turn_ids_updated_after_write():
    writer, mg, pg, _ = _make_writer()
    writer.write_turn("s1", "d1", _turn(0))
    assert writer._prev_turn_ids["d1"] == "t0"
    writer.write_turn("s1", "d1", _turn(1))
    assert writer._prev_turn_ids["d1"] == "t1"
    mg.write_turn_sequence.assert_called_once_with(prev_turn_id="t0", next_turn_id="t1")


# --- Cypher-level regression tests for critical bug fixes ---

import inspect

from graphknows.retrieval.stores.memgraph import MemgraphStore


def test_write_entity_node_sets_type_not_entity_type():
    src = inspect.getsource(MemgraphStore.write_entity_node)
    assert "e.type" in src and "e.entity_type" not in src


def test_write_file_node_uses_in_session_rel():
    src = inspect.getsource(MemgraphStore.write_file_node)
    assert "IN_SESSION" in src and "HAS_FILE" not in src


def test_write_turn_node_stores_session_id():
    sig = inspect.signature(MemgraphStore.write_turn_node)
    assert "session_id" in sig.parameters
    src = inspect.getsource(MemgraphStore.write_turn_node)
    assert "session_id" in src
