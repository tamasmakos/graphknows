# Brief: t_272d1d84 — write_chunk_node missing chunk_index as top-level property

## Goal
Fix `MemgraphStore.write_chunk_node` so it writes `chunk_index` as a top-level CHUNK node
property, not buried inside the serialised `metadata` string.

## Root Cause (already confirmed)
File: `graphknows/retrieval/stores/memgraph.py`, function `write_chunk_node` (~line 121).

The Cypher currently does:
```
SET c.metadata = $metadata
```
with `metadata=str(metadata)` — a Python string representation of the dict.

But `read_session_subgraph` (line ~650 of the same file) queries:
```cypher
c.chunk_index AS chunk_index
```
Since `chunk_index` was never written as a top-level property, this always returns `None`.

## Change 1 — Write failing test first (RED)

File: `tests/test_memgraph_store.py`

Add a new test class `TestWriteChunkNodeChunkIndex` with one test:
```python
@pytest.mark.asyncio
async def test_write_chunk_node_sets_chunk_index_property():
    """chunk_index must be a top-level CHUNK property, not buried in metadata."""
    store = MemgraphStore.__new__(MemgraphStore)
    calls = []

    class FakeSession:
        def run(self, cypher, **params):
            calls.append((cypher, params))
            class R:
                def consume(self): return None
            return R()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class FakeDriver:
        def session(self): return FakeSession()

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
```

Run the test — expect FAILURE (RED):
```bash
cd /root/graphknows && uv run pytest tests/test_memgraph_store.py::TestWriteChunkNodeChunkIndex -v 2>&1 | tail -20
```

## Change 2 — Fix production code (GREEN)

In `graphknows/retrieval/stores/memgraph.py`, update `write_chunk_node._sync`:

1. Extract `chunk_index` from the metadata dict before serialising it:
   ```python
   chunk_index = metadata.get("chunk_index")
   ```

2. Add `SET c.chunk_index = $chunk_index` to the Cypher:
   ```cypher
   MERGE (c:CHUNK {id: $cid})
   SET c.session_id = $sid,
       c.file_id = $fid,
       c.text = $text,
       c.heading_path = $heading_path,
       c.chunk_index = $chunk_index,
       c.metadata = $metadata
   ```

3. Pass `chunk_index=chunk_index` as a named parameter to `s.run(...)`.

Keep `c.metadata = $metadata` (still useful for other fields), but change the value passed
to `metadata=str(metadata)` → can keep as-is (it's the serialised dict for backward compat).

## Acceptance Criteria
- `TestWriteChunkNodeChunkIndex::test_write_chunk_node_sets_chunk_index_property` passes (GREEN)
- Full unit suite still passes: `uv run pytest tests/ -q -m 'not integration' --tb=no 2>&1 | tail -5`
- `uv run ruff check --config ruff.toml graphknows/` — clean
- `uv run ruff format --check --config ruff.toml graphknows/` — clean

## Files to Modify
- `tests/test_memgraph_store.py` — add `TestWriteChunkNodeChunkIndex`
- `graphknows/retrieval/stores/memgraph.py` — `write_chunk_node._sync`: add `chunk_index` extraction and parameter

## What NOT to do
- Do NOT modify `read_session_subgraph` or any other method
- Do NOT refactor the metadata serialisation beyond the minimal change
- Do NOT touch any other test files
- Do NOT add noqa comments
- Do NOT commit
