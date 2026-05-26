"""Unit tests for MemgraphStore."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from graphknows.retrieval.stores.memgraph import MemgraphStore


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_store_with_mock_driver() -> tuple[MemgraphStore, MagicMock, MagicMock]:
    store = MemgraphStore.__new__(MemgraphStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    return store, mock_driver, mock_session


def _make_store():
    """MemgraphStore with mocked neo4j Bolt driver."""
    store = MemgraphStore.__new__(MemgraphStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    return store, mock_driver, mock_session


# ---------------------------------------------------------------------------
# Module-level tests (from test_memgraph_store.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

def test_close_calls_driver_close():
    store, mock_driver, _ = _make_store()
    store.close()
    mock_driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_indexes_runs_index_stmts():
    store, _, mock_session = _make_store()
    await store.ensure_indexes()
    assert mock_session.run.called


@pytest.mark.asyncio
async def test_ensure_indexes_ignores_exceptions():
    """Index creation errors must be swallowed, not raised."""
    store, _, mock_session = _make_store()
    mock_session.run.side_effect = Exception("Index already exists")
    await store.ensure_indexes()


# ---------------------------------------------------------------------------
# write_session_metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_session_metadata_cypher():
    store, _, mock_session = _make_store()
    await store.write_session_metadata("s42", {"source": "test", "version": 1})
    mock_session.run.assert_called_once()
    cypher, = mock_session.run.call_args[0][:1]
    assert "MERGE" in cypher
    assert "SESSION" in cypher
    kwargs = mock_session.run.call_args[1]
    assert kwargs["sid"] == "s42"
    assert kwargs["props"] == {"source": "test", "version": 1}


# ---------------------------------------------------------------------------
# write_file_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_file_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_file_node(
        session_id="s1", file_id="f1", title="Doc A",
        parser_name="markdown", source_hint="file.md"
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MERGE" in cypher
    assert "FILE" in cypher
    assert "IN_SESSION" in cypher
    assert kwargs["fid"] == "f1"
    assert kwargs["sid"] == "s1"
    assert kwargs["title"] == "Doc A"
    assert kwargs["parser_name"] == "markdown"
    assert kwargs["source_hint"] == "file.md"


# ---------------------------------------------------------------------------
# write_chunk_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_chunk_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_chunk_node(
        session_id="s1", file_id="f1", chunk_id="c1",
        text="hello world", heading_path="H1 > H2",
        metadata={"idx": 0}
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "CHUNK" in cypher
    assert "HAS_CHUNK" in cypher
    assert kwargs["cid"] == "c1"
    assert kwargs["sid"] == "s1"
    assert kwargs["fid"] == "f1"
    assert kwargs["text"] == "hello world"
    assert kwargs["heading_path"] == "H1 > H2"
    assert "idx" in kwargs["metadata"]


# ---------------------------------------------------------------------------
# write_entity_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_entity_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_entity_node(
        session_id="s1", entity_name="Alice", entity_type="PERSON", confidence=0.9
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "ENTITY" in cypher
    assert "MERGE" in cypher
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"
    assert kwargs["etype"] == "PERSON"
    assert kwargs["conf"] == 0.9


@pytest.mark.asyncio
async def test_write_entity_node_confidence_max_logic():
    """Cypher must include CASE logic to keep higher confidence value."""
    store, _, mock_session = _make_store()
    await store.write_entity_node("s1", "Bob", "PERSON", 0.5)
    cypher = mock_session.run.call_args[0][0]
    assert "CASE" in cypher
    assert "confidence" in cypher


# ---------------------------------------------------------------------------
# write_chunk_entity_edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_chunk_entity_edge_cypher():
    store, _, mock_session = _make_store()
    await store.write_chunk_entity_edge(chunk_id="c1", entity_name="Alice", session_id="s1")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MENTIONS" in cypher
    assert "CHUNK" in cypher
    assert "ENTITY" in cypher
    assert kwargs["cid"] == "c1"
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"


# ---------------------------------------------------------------------------
# write_dialog_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_dialog_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_dialog_node("s1", "dlg1", "2026-01-01T00:00:00")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "DIALOG" in cypher
    assert "IN_SESSION" in cypher
    assert kwargs["did"] == "dlg1"
    assert kwargs["sid"] == "s1"
    assert kwargs["started_at"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# write_turn_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_node(
        dialog_id="dlg1", session_id="s1", turn_id="t1",
        turn_index=2, role="user", text="hello", timestamp="2026-01-01T00:01:00"
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TURN" in cypher
    assert "HAS_TURN" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["sid"] == "s1"
    assert kwargs["did"] == "dlg1"
    assert kwargs["idx"] == 2
    assert kwargs["role"] == "user"
    assert kwargs["text"] == "hello"


# ---------------------------------------------------------------------------
# write_turn_sequence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_sequence_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_sequence("t1", "t2")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "NEXT" in cypher
    assert kwargs["aid"] == "t1"
    assert kwargs["bid"] == "t2"


# ---------------------------------------------------------------------------
# write_turn_entity_edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_entity_edge_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_entity_edge("t1", "Alice", "s1", 0.85)
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TURN" in cypher
    assert "ENTITY" in cypher
    assert "MENTIONS" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"
    assert kwargs["conf"] == 0.85


# ---------------------------------------------------------------------------
# write_tool_call_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_tool_call_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_tool_call_node("t1", "search_web", "abc123", 250)
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TOOL_CALL" in cypher
    assert "CALLED" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["hash"] == "abc123"
    assert kwargs["name"] == "search_web"
    assert kwargs["latency_ms"] == 250


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_cypher():
    store, _, mock_session = _make_store()
    await store.delete_session("s99")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "DETACH DELETE" in cypher
    assert kwargs["sid"] == "s99"


# ---------------------------------------------------------------------------
# write_entities (deprecated batch method)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_entities_empty_list_is_noop():
    store, _, mock_session = _make_store()
    await store.write_entities("s1", [])
    mock_session.run.assert_not_called()


@pytest.mark.asyncio
async def test_write_entities_batch():
    store, _, mock_session = _make_store()

    class FakeEntity:
        name = "Python"
        label = "LANG"
        id = "e1"
        confidence = 0.9
        doc_ids = ["d1"]

    await store.write_entities("s1", [FakeEntity(), FakeEntity()])
    assert mock_session.run.call_count == 2
    cypher = mock_session.run.call_args_list[0][0][0]
    assert "Entity" in cypher
    assert "MERGE" in cypher


# ---------------------------------------------------------------------------
# write_relations (deprecated batch method)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_relations_empty_list_is_noop():
    store, _, mock_session = _make_store()
    await store.write_relations("s1", [])
    mock_session.run.assert_not_called()


@pytest.mark.asyncio
async def test_write_relations_batch():
    store, _, mock_session = _make_store()

    class FakeRelation:
        source = "e1"
        target = "e2"
        type = "RELATES_TO"

    await store.write_relations("s1", [FakeRelation()])
    cypher = mock_session.run.call_args[0][0]
    assert "RELATES" in cypher


# ---------------------------------------------------------------------------
# query_entity_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_entity_chunks_returns_parsed_rows():
    store, _, mock_session = _make_store()
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"chunk_id": "c1", "text": "hello"}[k]
    mock_session.run.return_value = [fake_row]

    results = await store.query_entity_chunks("s1", ["Alice", "Bob"], top_k=3)

    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MENTIONS" in cypher
    assert kwargs["names"] == ["Alice", "Bob"]
    assert kwargs["sid"] == "s1"
    assert kwargs["k"] == 3
    assert results == [{"chunk_id": "c1", "text": "hello"}]


# ---------------------------------------------------------------------------
# get_session_nodes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_nodes_returns_props():
    store, _, mock_session = _make_store()
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"name": "Alice", "type": "PERSON"}
    mock_session.run.return_value = [fake_row]

    nodes = await store.get_session_nodes("s1")
    assert len(nodes) == 1
    assert nodes[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# get_session_edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_edges_returns_edge_dicts():
    store, _, mock_session = _make_store()
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"from": "Alice", "to": "Bob", "rel_type": "KNOWS"}[k]
    mock_session.run.return_value = [fake_row]

    edges = await store.get_session_edges("s1")
    assert len(edges) == 1
    assert edges[0] == {"from": "Alice", "to": "Bob", "rel_type": "KNOWS"}


# ---------------------------------------------------------------------------
# get_session_graph (assembles nodes + edges)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_graph_structure():
    store, _, mock_session = _make_store()
    call_count = [0]

    def run_side(cypher, **kw):
        call_count[0] += 1
        if "properties(n)" in cypher:
            fake_row = MagicMock()
            fake_row.__getitem__ = lambda self, k: {"name": "Alice"}
            return [fake_row]
        if "RELATES" in cypher:
            fake_row = MagicMock()
            fake_row.__getitem__ = lambda self, k: {"from": "A", "to": "B", "rel_type": "X"}[k]
            return [fake_row]
        return []

    mock_session.run.side_effect = run_side

    graph = await store.get_session_graph("s1")
    assert graph["session_id"] == "s1"
    assert "node_count" in graph
    assert "edge_count" in graph
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)


# ---------------------------------------------------------------------------
# TestWriteChunkNodeChunkIndex
# ---------------------------------------------------------------------------

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
        assert "chunk_index" in cypher.lower(), "Cypher must SET c.chunk_index"
        assert "$chunk_index" in cypher or "chunk_index" in params, \
            "chunk_index param must be passed separately"
        assert params.get("chunk_index") == 3, f"Expected chunk_index=3, got {params}"


# ---------------------------------------------------------------------------
# TestWriteEntityNodeNullType
# ---------------------------------------------------------------------------

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

        await store.write_entity_node(
            session_id="s1",
            entity_name="Barack Obama",
            entity_type=None,
            confidence=0.9,
        )

        assert calls, "run() was never called"
        _cypher, params = calls[0]
        assert params["etype"] is not None, "etype must not be None"
        assert params["etype"] == "ENTITY", f"Expected default 'ENTITY', got {params['etype']!r}"


# ===========================================================================
# Extended memgraph store tests
# ===========================================================================


from unittest.mock import MagicMock

import pytest

from graphknows.retrieval.stores.memgraph import MemgraphStore


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_store():
    """MemgraphStore with mocked neo4j Bolt driver."""
    store = MemgraphStore.__new__(MemgraphStore)
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store._driver = mock_driver
    return store, mock_driver, mock_session


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

def test_close_calls_driver_close():
    store, mock_driver, _ = _make_store()
    store.close()
    mock_driver.close.assert_called_once()


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_indexes_runs_index_stmts():
    store, _, mock_session = _make_store()
    await store.ensure_indexes()
    # Should have called s.run() at least once per index in MEMGRAPH_INDEXES
    assert mock_session.run.called


@pytest.mark.asyncio
async def test_ensure_indexes_ignores_exceptions():
    """Index creation errors must be swallowed, not raised."""
    store, _, mock_session = _make_store()
    mock_session.run.side_effect = Exception("Index already exists")
    # Must not raise
    await store.ensure_indexes()


# ---------------------------------------------------------------------------
# write_session_metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_session_metadata_cypher():
    store, _, mock_session = _make_store()
    await store.write_session_metadata("s42", {"source": "test", "version": 1})
    mock_session.run.assert_called_once()
    cypher, = mock_session.run.call_args[0][:1]
    assert "MERGE" in cypher
    assert "SESSION" in cypher
    kwargs = mock_session.run.call_args[1]
    assert kwargs["sid"] == "s42"
    assert kwargs["props"] == {"source": "test", "version": 1}


# ---------------------------------------------------------------------------
# write_file_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_file_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_file_node(
        session_id="s1", file_id="f1", title="Doc A",
        parser_name="markdown", source_hint="file.md"
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MERGE" in cypher
    assert "FILE" in cypher
    assert "IN_SESSION" in cypher
    assert kwargs["fid"] == "f1"
    assert kwargs["sid"] == "s1"
    assert kwargs["title"] == "Doc A"
    assert kwargs["parser_name"] == "markdown"
    assert kwargs["source_hint"] == "file.md"


# ---------------------------------------------------------------------------
# write_chunk_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_chunk_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_chunk_node(
        session_id="s1", file_id="f1", chunk_id="c1",
        text="hello world", heading_path="H1 > H2",
        metadata={"idx": 0}
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "CHUNK" in cypher
    assert "HAS_CHUNK" in cypher
    assert kwargs["cid"] == "c1"
    assert kwargs["sid"] == "s1"
    assert kwargs["fid"] == "f1"
    assert kwargs["text"] == "hello world"
    assert kwargs["heading_path"] == "H1 > H2"
    # metadata is serialized to string
    assert "idx" in kwargs["metadata"]


# ---------------------------------------------------------------------------
# write_entity_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_entity_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_entity_node(
        session_id="s1", entity_name="Alice", entity_type="PERSON", confidence=0.9
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "ENTITY" in cypher
    assert "MERGE" in cypher
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"
    assert kwargs["etype"] == "PERSON"
    assert kwargs["conf"] == 0.9


@pytest.mark.asyncio
async def test_write_entity_node_confidence_max_logic():
    """Cypher must include CASE logic to keep higher confidence value."""
    store, _, mock_session = _make_store()
    await store.write_entity_node("s1", "Bob", "PERSON", 0.5)
    cypher = mock_session.run.call_args[0][0]
    assert "CASE" in cypher
    assert "confidence" in cypher


# ---------------------------------------------------------------------------
# write_chunk_entity_edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_chunk_entity_edge_cypher():
    store, _, mock_session = _make_store()
    await store.write_chunk_entity_edge(chunk_id="c1", entity_name="Alice", session_id="s1")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MENTIONS" in cypher
    assert "CHUNK" in cypher
    assert "ENTITY" in cypher
    assert kwargs["cid"] == "c1"
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"


# ---------------------------------------------------------------------------
# write_dialog_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_dialog_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_dialog_node("s1", "dlg1", "2026-01-01T00:00:00")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "DIALOG" in cypher
    assert "IN_SESSION" in cypher
    assert kwargs["did"] == "dlg1"
    assert kwargs["sid"] == "s1"
    assert kwargs["started_at"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# write_turn_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_node(
        dialog_id="dlg1", session_id="s1", turn_id="t1",
        turn_index=2, role="user", text="hello", timestamp="2026-01-01T00:01:00"
    )
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TURN" in cypher
    assert "HAS_TURN" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["sid"] == "s1"
    assert kwargs["did"] == "dlg1"
    assert kwargs["idx"] == 2
    assert kwargs["role"] == "user"
    assert kwargs["text"] == "hello"


# ---------------------------------------------------------------------------
# write_turn_sequence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_sequence_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_sequence("t1", "t2")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "NEXT" in cypher
    assert kwargs["aid"] == "t1"
    assert kwargs["bid"] == "t2"


# ---------------------------------------------------------------------------
# write_turn_entity_edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_turn_entity_edge_cypher():
    store, _, mock_session = _make_store()
    await store.write_turn_entity_edge("t1", "Alice", "s1", 0.85)
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TURN" in cypher
    assert "ENTITY" in cypher
    assert "MENTIONS" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["name"] == "Alice"
    assert kwargs["sid"] == "s1"
    assert kwargs["conf"] == 0.85


# ---------------------------------------------------------------------------
# write_tool_call_node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_tool_call_node_cypher():
    store, _, mock_session = _make_store()
    await store.write_tool_call_node("t1", "search_web", "abc123", 250)
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "TOOL_CALL" in cypher
    assert "CALLED" in cypher
    assert kwargs["tid"] == "t1"
    assert kwargs["hash"] == "abc123"
    assert kwargs["name"] == "search_web"
    assert kwargs["latency_ms"] == 250


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_cypher():
    store, _, mock_session = _make_store()
    await store.delete_session("s99")
    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "DETACH DELETE" in cypher
    assert kwargs["sid"] == "s99"


# ---------------------------------------------------------------------------
# write_entities (deprecated batch method)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_entities_empty_list_is_noop():
    store, _, mock_session = _make_store()
    await store.write_entities("s1", [])
    mock_session.run.assert_not_called()


@pytest.mark.asyncio
async def test_write_entities_batch():
    store, _, mock_session = _make_store()

    class FakeEntity:
        name = "Python"
        label = "LANG"
        id = "e1"
        confidence = 0.9
        doc_ids = ["d1"]

    await store.write_entities("s1", [FakeEntity(), FakeEntity()])
    # Called once per entity in the list
    assert mock_session.run.call_count == 2
    # Each call should MERGE an Entity node
    cypher = mock_session.run.call_args_list[0][0][0]
    assert "Entity" in cypher
    assert "MERGE" in cypher


# ---------------------------------------------------------------------------
# write_relations (deprecated batch method)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_relations_empty_list_is_noop():
    store, _, mock_session = _make_store()
    await store.write_relations("s1", [])
    mock_session.run.assert_not_called()


@pytest.mark.asyncio
async def test_write_relations_batch():
    store, _, mock_session = _make_store()

    class FakeRelation:
        source = "e1"
        target = "e2"
        type = "RELATES_TO"

    await store.write_relations("s1", [FakeRelation()])
    cypher = mock_session.run.call_args[0][0]
    assert "RELATES" in cypher


# ---------------------------------------------------------------------------
# query_entity_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_entity_chunks_returns_parsed_rows():
    store, _, mock_session = _make_store()
    # Mock the result: s.run() returns an iterable of row-like objects
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"chunk_id": "c1", "text": "hello"}[k]
    mock_session.run.return_value = [fake_row]

    results = await store.query_entity_chunks("s1", ["Alice", "Bob"], top_k=3)

    cypher = mock_session.run.call_args[0][0]
    kwargs = mock_session.run.call_args[1]
    assert "MENTIONS" in cypher
    assert kwargs["names"] == ["Alice", "Bob"]
    assert kwargs["sid"] == "s1"
    assert kwargs["k"] == 3
    assert results == [{"chunk_id": "c1", "text": "hello"}]


# ---------------------------------------------------------------------------
# get_session_nodes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_nodes_returns_props():
    store, _, mock_session = _make_store()
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"name": "Alice", "type": "PERSON"}
    mock_session.run.return_value = [fake_row]

    nodes = await store.get_session_nodes("s1")
    assert len(nodes) == 1
    assert nodes[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# get_session_edges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_edges_returns_edge_dicts():
    store, _, mock_session = _make_store()
    fake_row = MagicMock()
    fake_row.__getitem__ = lambda self, k: {"from": "Alice", "to": "Bob", "rel_type": "KNOWS"}[k]
    mock_session.run.return_value = [fake_row]

    edges = await store.get_session_edges("s1")
    assert len(edges) == 1
    assert edges[0] == {"from": "Alice", "to": "Bob", "rel_type": "KNOWS"}


# ---------------------------------------------------------------------------
# get_session_graph (assembles nodes + edges)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_graph_structure():
    store, _, mock_session = _make_store()
    # nodes and edges both returned from different calls
    call_count = [0]
    def run_side(cypher, **kw):
        call_count[0] += 1
        if "properties(n)" in cypher:
            fake_row = MagicMock()
            fake_row.__getitem__ = lambda self, k: {"name": "Alice"}
            return [fake_row]
        if "RELATES" in cypher:
            fake_row = MagicMock()
            fake_row.__getitem__ = lambda self, k: {"from": "A", "to": "B", "rel_type": "X"}[k]
            return [fake_row]
        return []

    mock_session.run.side_effect = run_side

    graph = await store.get_session_graph("s1")
    assert graph["session_id"] == "s1"
    assert "node_count" in graph
    assert "edge_count" in graph
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
