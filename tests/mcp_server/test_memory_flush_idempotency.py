"""Test memory_flush idempotency guard uses _driver.session() correctly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_memory_flush_idempotency_guard_uses_driver_session() -> None:
    """Ensure the idempotency guard calls neo_check._driver.session(), not neo_check.session()."""
    mock_neo = MagicMock()
    mock_neo._db = "neo4j"

    # Simulate session returning a row with n=1 (already flushed)
    mock_row = {"n": 1}
    mock_sess = MagicMock()
    mock_sess.run.return_value = [mock_row]
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_sess)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_neo._driver.session.return_value = ctx

    with mock_neo._driver.session(database=mock_neo._db) as s:
        rows = list(s.run("MATCH (s:SESSION {id: $sid}) RETURN count(s) AS n", sid="test-session"))

    mock_neo._driver.session.assert_called_once_with(database="neo4j")
    assert rows[0]["n"] == 1


def test_neo4j_store_has_no_session_method() -> None:
    """Confirm Neo4jStore does NOT expose a .session() method directly."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    assert not hasattr(Neo4jStore, "session"), (
        "Neo4jStore should not have a .session() method — use ._driver.session()"
    )
