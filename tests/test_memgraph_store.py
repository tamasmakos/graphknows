"""Unit tests for MemgraphStore."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from graphknows.retrieval.stores.memgraph import MemgraphStore


def _make_store_with_mock_driver() -> tuple[MemgraphStore, MagicMock, MagicMock]:
    store = MemgraphStore.__new__(MemgraphStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    return store, mock_driver, mock_session


@pytest.mark.asyncio
async def test_write_relation_edge_generates_correct_cypher():
    store, mock_driver, mock_session = _make_store_with_mock_driver()
    now = datetime(2026, 1, 1)

    await store.write_relation_edge(
        session_id="s1",
        subject_name="Viktor Orbán",
        relation_name="OPPOSES",
        object_name="Ukraine EU membership",
        fact="Viktor Orbán opposes Ukraine EU membership",
        confidence=1.0,
        valid_at=now,
        created_at=now,
    )

    assert mock_session.run.called
    cypher, params = mock_session.run.call_args[0][0], mock_session.run.call_args[1]
    assert "MERGE" in cypher
    assert ":REL" in cypher
    assert "$name" in cypher or "name" in cypher
    assert "$fact" in cypher or "fact" in cypher
    assert params["session_id"] == "s1"
    assert params["subject"] == "Viktor Orbán"
    assert params["object"] == "Ukraine EU membership"
    assert params["name"] == "OPPOSES"
    assert params["fact"] == "Viktor Orbán opposes Ukraine EU membership"
    assert params["confidence"] == 1.0
