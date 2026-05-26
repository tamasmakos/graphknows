"""
Tests for PgVectorStore — using a fake asyncpg pool.

Strategy
--------
- Never touch a real Postgres. We create a `PgVectorStore` via `__new__` (skip __init__)
  and inject a `MagicMock` pool.
- Mock `pool.acquire()` as an async context manager that returns a fake `conn`.
- Assert SQL content (substring checks) and parameter correctness.
- Cover edge cases: empty inputs, missing optional fields, delete returns, error paths.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.retrieval.stores.pgvector import PgVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_row(**kw):
    """Dict-like object that supports r['key'] access (like asyncpg Record)."""
    d = dict(**kw)

    class _Row:
        def __getitem__(self, k):
            return d[k]
        def get(self, k, default=None):
            return d.get(k, default)

    return _Row()


def _make_store():
    """Build a PgVectorStore with a mocked asyncpg pool — no DB required."""
    store = PgVectorStore.__new__(PgVectorStore)
    store.dsn = "postgresql://fake/fake"

    fake_conn = AsyncMock()
    fake_conn.executemany = AsyncMock(return_value=None)
    fake_conn.execute = AsyncMock(return_value="DELETE 3")
    fake_conn.fetch = AsyncMock(return_value=[])

    @asynccontextmanager
    async def _acquire():
        yield fake_conn

    fake_pool = MagicMock()
    fake_pool.acquire = _acquire

    store._pool = fake_pool
    return store, fake_conn


# ---------------------------------------------------------------------------
# Lifecycle: _pool_or_raise
# ---------------------------------------------------------------------------

def test_pool_or_raise_when_connected():
    store, _ = _make_store()
    # No exception if pool is set
    assert store._pool_or_raise() is store._pool


def test_pool_or_raise_raises_when_not_connected():
    store = PgVectorStore.__new__(PgVectorStore)
    store._pool = None
    with pytest.raises(RuntimeError, match="not connected"):
        store._pool_or_raise()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_chunks_calls_executemany():
    store, conn = _make_store()
    chunks = [
        {"id": "c1", "doc_id": "d1", "text": "hello world", "embedding": [0.1, 0.2], "metadata": {"source": "test"}},
    ]
    await store.upsert_chunks("sess1", chunks)
    conn.executemany.assert_called_once()
    sql, records = conn.executemany.call_args[0]
    assert "INSERT INTO chunks" in sql
    assert "ON CONFLICT" in sql
    # record should be (id, session_id, doc_id, text_truncated, emb_json, meta_json)
    assert records[0][0] == "c1"
    assert records[0][1] == "sess1"
    assert records[0][2] == "d1"
    assert records[0][3] == "hello world"
    assert json.loads(records[0][4]) == [0.1, 0.2]


@pytest.mark.asyncio
async def test_upsert_chunks_empty_list_is_noop():
    """Empty input → no DB call at all."""
    store, conn = _make_store()
    await store.upsert_chunks("sess1", [])
    conn.executemany.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_chunks_truncates_long_text():
    """Text longer than 4000 chars should be truncated."""
    store, conn = _make_store()
    long_text = "x" * 5000
    await store.upsert_chunks("s", [{"id": "c1", "doc_id": "d", "text": long_text, "embedding": []}])
    _, records = conn.executemany.call_args[0]
    assert len(records[0][3]) == 4000


@pytest.mark.asyncio
async def test_upsert_chunks_default_metadata():
    """Missing metadata defaults to empty JSON object."""
    store, conn = _make_store()
    await store.upsert_chunks("s", [{"id": "c1", "doc_id": "d", "text": "t", "embedding": [0.0]}])
    _, records = conn.executemany.call_args[0]
    assert records[0][5] == "{}"


# ---------------------------------------------------------------------------
# upsert_entities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_entities_empty_list_is_noop():
    store, conn = _make_store()
    await store.upsert_entities([])
    conn.executemany.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_entities_sql_content():
    store, conn = _make_store()
    await store.upsert_entities([
        {"name": "python", "label": "LANG", "embedding": [0.5, 0.6], "chunk_count": 3}
    ])
    sql, records = conn.executemany.call_args[0]
    assert "INSERT INTO entities" in sql
    assert "ON CONFLICT" in sql
    assert records[0][0] == "python"
    assert records[0][1] == "LANG"
    assert json.loads(records[0][2]) == [0.5, 0.6]
    assert records[0][3] == 3


@pytest.mark.asyncio
async def test_upsert_entities_default_label():
    """Missing label defaults to 'Entity'."""
    store, conn = _make_store()
    await store.upsert_entities([{"name": "foo", "embedding": [0.1]}])
    _, records = conn.executemany.call_args[0]
    assert records[0][1] == "Entity"


@pytest.mark.asyncio
async def test_upsert_entities_default_chunk_count():
    """Missing chunk_count defaults to 0."""
    store, conn = _make_store()
    await store.upsert_entities([{"name": "foo", "label": "X", "embedding": [0.1]}])
    _, records = conn.executemany.call_args[0]
    assert records[0][3] == 0


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_chunks_returns_parsed_rows():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[
        _fake_row(id="c1", doc_id="d1", text="hello", metadata=json.dumps({"k": "v"}), cosine_score=0.88),
    ])
    results = await store.search_chunks([0.1, 0.2], session_id="s1", k=5)
    assert len(results) == 1
    r = results[0]
    assert r["chunk_id"] == "c1"
    assert r["doc_id"] == "d1"
    assert r["text"] == "hello"
    assert r["metadata"] == {"k": "v"}
    assert abs(r["cosine_score"] - 0.88) < 1e-6


@pytest.mark.asyncio
async def test_search_chunks_empty_result():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[])
    results = await store.search_chunks([0.1], session_id="s99", k=10)
    assert results == []


@pytest.mark.asyncio
async def test_search_chunks_null_doc_id_defaults_empty_string():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[
        _fake_row(id="c1", doc_id=None, text="t", metadata=None, cosine_score=0.5),
    ])
    results = await store.search_chunks([0.1], session_id="s1", k=1)
    assert results[0]["doc_id"] == ""


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_entities_no_filter():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[
        _fake_row(name="alice", label="PERSON", score=0.9),
    ])
    results = await store.search_entities([0.1], k=5)
    assert len(results) == 1
    assert results[0]["name"] == "alice"
    assert results[0]["label"] == "PERSON"
    sql = conn.fetch.call_args[0][0]
    # No WHERE clause with label filter
    assert "ANY" not in sql


@pytest.mark.asyncio
async def test_search_entities_with_type_filter():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[])
    await store.search_entities([0.1], k=5, entity_types=["ORG", "PERSON"])
    sql = conn.fetch.call_args[0][0]
    assert "ANY" in sql


@pytest.mark.asyncio
async def test_search_entities_null_label_defaults():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[
        _fake_row(name="x", label=None, score=0.5),
    ])
    results = await store.search_entities([0.1])
    assert results[0]["label"] == "Entity"


# ---------------------------------------------------------------------------
# get_chunks_by_ids
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_chunks_by_ids_empty():
    store, conn = _make_store()
    result = await store.get_chunks_by_ids([])
    assert result == {}
    conn.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_get_chunks_by_ids_returns_dict():
    store, conn = _make_store()
    conn.fetch = AsyncMock(return_value=[
        _fake_row(id="c1", doc_id="d1", text="text", metadata=json.dumps({"x": 1})),
        _fake_row(id="c2", doc_id="d2", text="other", metadata=None),
    ])
    result = await store.get_chunks_by_ids(["c1", "c2"])
    assert set(result.keys()) == {"c1", "c2"}
    assert result["c1"]["text"] == "text"
    assert result["c1"]["metadata"] == {"x": 1}
    assert result["c2"]["doc_id"] == "d2"
    assert result["c2"]["metadata"] == {}


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_returns_count():
    store, conn = _make_store()
    conn.execute = AsyncMock(return_value="DELETE 5")
    count = await store.delete_session("sess42")
    assert count == 5
    sql = conn.execute.call_args[0][0]
    assert "DELETE FROM chunks" in sql


@pytest.mark.asyncio
async def test_delete_session_zero_rows():
    store, conn = _make_store()
    conn.execute = AsyncMock(return_value="DELETE 0")
    count = await store.delete_session("empty_sess")
    assert count == 0


# ---------------------------------------------------------------------------
# close() + connect() lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_clears_pool():
    store, _ = _make_store()
    mock_close = AsyncMock()
    store._pool.close = mock_close
    pool_ref = store._pool  # capture before close sets it to None
    await store.close()
    mock_close.assert_called_once()
    assert store._pool is None


@pytest.mark.asyncio
async def test_close_noop_if_not_connected():
    store = PgVectorStore.__new__(PgVectorStore)
    store._pool = None
    # Must not raise
    await store.close()


# ---------------------------------------------------------------------------
# write_stm_chunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_stm_chunk_sql_and_params():
    store, conn = _make_store()
    chunk = {
        "id": "stm1", "session_id": "s1", "text": "stm text",
        "file_id": "f1", "heading_path": "H1 > H2",
        "embedding": [0.1, 0.2], "metadata": {"a": 1},
    }
    await store.write_stm_chunk(chunk)
    conn.execute.assert_called_once()
    sql, *params = conn.execute.call_args[0]
    assert "stm_chunks" in sql
    assert params[0] == "stm1"
    assert params[1] == "s1"
    assert params[3] == "stm text"
    assert json.loads(params[5]) == [0.1, 0.2]


@pytest.mark.asyncio
async def test_write_stm_chunk_no_embedding():
    """Missing embedding should pass None to the DB."""
    store, conn = _make_store()
    chunk = {"id": "s1", "session_id": "sess", "text": "hello"}
    await store.write_stm_chunk(chunk)
    sql, *params = conn.execute.call_args[0]
    assert params[5] is None  # embedding param


# ---------------------------------------------------------------------------
# write_stm_turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_stm_turn_sql_and_params():
    store, conn = _make_store()
    turn = {
        "id": "t1", "session_id": "s1", "role": "user", "text": "hello",
        "dialog_id": "dlg1", "turn_index": 3, "embedding": [0.3],
    }
    await store.write_stm_turn(turn)
    sql, *params = conn.execute.call_args[0]
    assert "stm_turns" in sql
    assert params[4] == "user"
    assert params[5] == "hello"
    assert json.loads(params[6]) == [0.3]


# ---------------------------------------------------------------------------
# promote_to_ltm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_to_ltm_returns_counts():
    store, conn = _make_store()
    call_count = [0]
    async def execute_side(*args, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return "INSERT 0 7"  # chunks
        return "INSERT 0 3"  # turns

    conn.execute = AsyncMock(side_effect=execute_side)
    chunks, turns = await store.promote_to_ltm("s1")
    assert chunks == 7
    assert turns == 3
