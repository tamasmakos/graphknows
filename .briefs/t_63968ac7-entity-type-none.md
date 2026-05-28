# Brief: t_63968ac7 — ENTITY.type None when GLiNER2 returns explicit None

## Goal
Fix `MemgraphStore.write_entity_node` to guard against `entity_type=None` being written
as a null property, by normalising it to a sensible default.

## Root Cause (already confirmed)
File: `graphknows/retrieval/stores/memgraph.py`, function `write_entity_node` (~line 169).

The Cypher does: `SET e.type = $etype`
The callers use `ent.get("type", "ENTITY")` which returns the default only when the key is
**absent**. But GLiNER2 sometimes returns `{"name": "foo", "type": None}` — the key is
present, value is `None`. So `ent.get("type", "ENTITY")` returns `None`, and `None` is
passed as `etype`, writing `e.type = null` to the graph.

## Change 1 — Write failing test first (RED)

File: `tests/test_memgraph_store.py`

Add new test class `TestWriteEntityNodeNullType`:
```python
@pytest.mark.asyncio
async def test_write_entity_node_normalises_none_type():
    """entity_type=None must be written as 'ENTITY', not null."""
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

    # Simulate GLiNER2 returning explicit None for type
    await store.write_entity_node(
        session_id="s1",
        entity_name="Barack Obama",
        entity_type=None,   # <-- explicit None, not missing
        confidence=0.9,
    )

    assert calls, "run() was never called"
    _cypher, params = calls[0]
    assert params["etype"] is not None, "etype must not be None"
    assert params["etype"] == "ENTITY", f"Expected default 'ENTITY', got {params['etype']!r}"
```

Run — expect FAILURE (RED):
```bash
cd /root/graphknows && uv run pytest tests/test_memgraph_store.py::TestWriteEntityNodeNullType -v 2>&1 | tail -20
```

## Change 2 — Fix production code (GREEN)

In `graphknows/retrieval/stores/memgraph.py`, function `write_entity_node`:

Change the method signature or the first line of `_sync` to normalise:
```python
async def write_entity_node(
    self,
    session_id: str,
    entity_name: str,
    entity_type: str | None,   # accept None explicitly
    confidence: float,
) -> None:
    def _sync() -> None:
        etype = entity_type or "ENTITY"   # normalise None → "ENTITY"
        with self._driver.session() as s:
            s.run(
                ...,
                etype=etype,
                ...
            ).consume()
```

The simplest fix: `etype = entity_type or "ENTITY"` at the start of `_sync`.

## Acceptance Criteria
- `TestWriteEntityNodeNullType::test_write_entity_node_normalises_none_type` passes (GREEN)
- Full unit suite still passes: `uv run pytest tests/ -q -m 'not integration' --tb=no 2>&1 | tail -5`
- `uv run ruff check --config ruff.toml graphknows/` — clean
- `uv run ruff format --check --config ruff.toml graphknows/` — clean

## Files to Modify
- `tests/test_memgraph_store.py` — add `TestWriteEntityNodeNullType`
- `graphknows/retrieval/stores/memgraph.py` — `write_entity_node`: normalise None type

## What NOT to do
- Do NOT change the signature of callers (ingestor.py, dialog.py)
- Do NOT modify any other method
- Do NOT touch any other test files
- Do NOT add noqa comments
- Do NOT commit
