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


class TestWriteChunkNodeChunkIndex:
    @pytest.mark.asyncio
    async def test_write_chunk_node_sets_chunk_index_property(self):
        """chunk_index must be a top-level CHUNK property, not buried in metadata."""
        store = MemgraphStore.__new__(MemgraphStore)
        calls = []

        class FakeSession:
            def run(self, cypher, **params):
                calls.append((cypher, params))

                class R:
                    def consume(self):
                        return None

                return R()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        class FakeDriver:
            def session(self):
                return FakeSession()

        store._driver = FakeDriver()

        await store.write_chunk_node(
            session_id="s1",
            file_id="f1",
            chunk_id="c1",
            text="hello world",
            heading_path="h/p",
            metadata={"chunk_index": 3, "extra": "val"},
        )

        assert calls, "run() was never called"
        cypher, params = calls[0]
        # chunk_index must be a top-level property set via parameter
        assert "chunk_index" in cypher.lower(), "Cypher must SET c.chunk_index"
        assert "$chunk_index" in cypher or "chunk_index" in params, \
            "chunk_index param must be passed separately"
        assert params.get("chunk_index") == 3, f"Expected chunk_index=3, got {params}"


class TestWriteEntityNodeNullType:
    @pytest.mark.asyncio
    async def test_write_entity_node_normalises_none_type(self):
        """entity_type=None must be written as 'ENTITY', not null."""
        store = MemgraphStore.__new__(MemgraphStore)
        calls = []

        class FakeSession:
            def run(self, cypher, **params):
                calls.append((cypher, params))

                class R:
                    def consume(self):
                        return None

                return R()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        class FakeDriver:
            def session(self):
                return FakeSession()

        store._driver = FakeDriver()

        # Simulate GLiNER2 returning explicit None for type
        await store.write_entity_node(
            session_id="s1",
            entity_name="Barack Obama",
            entity_type=None,  # <-- explicit None, not missing
            confidence=0.9,
        )

        assert calls, "run() was never called"
        _cypher, params = calls[0]
        assert params["etype"] is not None, "etype must not be None"
        assert params["etype"] == "ENTITY", f"Expected default 'ENTITY', got {params['etype']!r}"
