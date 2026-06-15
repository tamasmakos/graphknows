"""Unit tests for graph schema application and PgVectorStore DDL methods."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from graphknows.memory.stores.pgvector import PgVectorStore
from graphknows.ingestion.domain.schema import apply_memgraph_schema, apply_schema

# ---------------------------------------------------------------------------
# Neo4j schema
# ---------------------------------------------------------------------------


def test_apply_schema_is_idempotent():
    """Calling apply_schema twice on the same mock session produces no errors."""
    mock_session = MagicMock()

    result1 = apply_schema(mock_session)
    result2 = apply_schema(mock_session)

    assert result1["errors"] == []
    assert result2["errors"] == []
    assert result1["constraints"] == result2["constraints"]
    assert result1["indexes"] == result2["indexes"]
    assert mock_session.run.call_count == (result1["constraints"] + result1["indexes"]) * 2


def test_apply_schema_counts_constraints_and_indexes():
    """apply_schema returns positive counts for constraints and indexes."""
    mock_session = MagicMock()

    result = apply_schema(mock_session)

    assert result["constraints"] > 0
    assert result["indexes"] > 0
    assert result["errors"] == []


def test_apply_schema_logs_on_exception():
    """apply_schema captures errors when session.run raises without propagating."""
    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("boom")

    result = apply_schema(mock_session)

    assert len(result["errors"]) == 12  # 7 constraints + 5 indexes
    assert result["constraints"] == 0
    assert result["indexes"] == 0
    assert all("boom" in err for err in result["errors"])


# ---------------------------------------------------------------------------
# Memgraph schema
# ---------------------------------------------------------------------------


def test_apply_memgraph_schema_is_idempotent():
    """Calling apply_memgraph_schema twice on the same mock session produces no errors."""
    mock_session = MagicMock()
    mock_session.run.return_value.consume.return_value = None

    result1 = apply_memgraph_schema(mock_session)
    result2 = apply_memgraph_schema(mock_session)

    assert result1["errors"] == []
    assert result2["errors"] == []
    assert result1["indexes"] == result2["indexes"]
    assert mock_session.run.call_count == result1["indexes"] * 2


def test_apply_memgraph_schema_counts_indexes():
    """apply_memgraph_schema returns positive count for indexes."""
    mock_session = MagicMock()
    mock_session.run.return_value.consume.return_value = None

    result = apply_memgraph_schema(mock_session)

    assert result["indexes"] > 0
    assert result["errors"] == []


def test_apply_memgraph_schema_logs_on_exception():
    """apply_memgraph_schema captures errors when session.run raises."""
    mock_session = MagicMock()
    mock_session.run.return_value.consume.side_effect = RuntimeError("memgraph err")

    result = apply_memgraph_schema(mock_session)

    assert len(result["errors"]) == 7  # 7 memgraph indexes
    assert result["indexes"] == 0
    assert all("memgraph err" in err for err in result["errors"])


# ---------------------------------------------------------------------------
# PgVectorStore method signatures
# ---------------------------------------------------------------------------


def test_pgvector_store_has_required_stm_methods():
    """PgVectorStore exposes all STM/LTM write and search methods."""
    required = {
        "write_stm_chunk": {"chunk"},
        "write_stm_turn": {"turn"},
        "search_stm_chunks": {"session_id", "embedding", "top_k"},
        "search_stm_turns": {"session_id", "embedding", "top_k"},
        "promote_to_ltm": {"session_id"},
        "clear_stm": {"session_id"},
        "write_ltm_topic": {"topic"},
        "write_community_assignments": {"session_id", "assignments"},
    }

    for method_name, expected_params in required.items():
        assert hasattr(PgVectorStore, method_name), f"PgVectorStore missing {method_name}"
        sig = inspect.signature(getattr(PgVectorStore, method_name))
        params = set(sig.parameters.keys()) - {"self"}
        assert params == expected_params, (
            f"{method_name} params {params} != expected {expected_params}"
        )
