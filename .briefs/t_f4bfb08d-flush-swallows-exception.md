# Brief: t_f4bfb08d — flush.py swallows Neo4j write exceptions silently

## Goal
Fix `SessionFlusher._write_ltm_graph` in `graphknows/memory/flush.py` so that when
`write_session_subgraph` raises an exception, the error is surfaced in `FlushResult.errors`
and `nodes`/`edges` stay at 0 — not silently reported as success.

## Root Cause (already confirmed)
File: `graphknows/memory/flush.py`, method `_write_ltm_graph` (~lines 180–188).

Current code:
```python
try:
    result = await self._neo.write_session_subgraph(session_id, stm_data)
    nodes = result.get("nodes", 0)
    edges = result.get("edges", 0)
except Exception as exc:
    log.warning("write_ltm_graph %s: %s", session_id, exc)   # ← swallows error
# then continues: edges += rel_edges, return {"nodes": nodes, "edges": edges}
```

After the exception, `nodes` and `edges` are unset (NameError) **or** remain 0, and the
exception is never surfaced to the caller. The `flush()` method that calls this never sees
an error and `FlushResult.errors` stays empty — caller thinks the flush succeeded.

Look at how `_write_ltm_graph` returns and how its caller uses the result. The fix must
propagate the error into `FlushResult.errors` (the standard error-collection list used
throughout `flush.py`).

## Change 1 — Write failing test first (RED)

File: `tests/test_flush.py`

Add test class `TestFlushSurfacesNeo4jError`:
```python
from unittest.mock import AsyncMock, MagicMock
from graphknows.memory.flush import SessionFlusher, FlushResult

class TestFlushSurfacesNeo4jError:
    @pytest.mark.asyncio
    async def test_neo4j_write_error_appears_in_flush_result_errors(self):
        """When write_session_subgraph raises, FlushResult.errors must contain the error."""
        mg = MagicMock()
        pg = MagicMock()
        neo = AsyncMock()

        # Force Neo4j write to raise
        neo.write_session_subgraph = AsyncMock(
            side_effect=RuntimeError("neo4j connection refused")
        )
        neo.extract_and_write_relations = AsyncMock(return_value=0)

        flusher = SessionFlusher.__new__(SessionFlusher)
        flusher._mg = mg
        flusher._pg = pg
        flusher._neo = neo
        flusher._openrouter_key = None

        # Call the private helper directly or call flush() with mocked STM data
        # The simplest approach: call _write_ltm_graph directly
        errors: list[str] = []
        result = await flusher._write_ltm_graph(
            session_id="test-session",
            stm_data={"entities": [], "chunks": [], "turns": []},
            errors=errors,
        )

        # The error must be surfaced
        assert errors, "errors list must be populated when Neo4j raises"
        assert any("neo4j" in e.lower() or "connection refused" in e.lower() for e in errors), \
            f"Expected Neo4j error in errors list, got: {errors}"
```

Note: look at the actual signature of `_write_ltm_graph` in `flush.py` before writing
the test. If it does NOT accept an `errors` list argument, adjust the test to call the
`flush()` public method instead and inspect `FlushResult.errors`. The key assertion is:
**the Neo4j write error must appear somewhere in the returned FlushResult or errors list**.

Run — expect FAILURE (RED):
```bash
cd /root/graphknows && uv run pytest tests/test_flush.py::TestFlushSurfacesNeo4jError -v 2>&1 | tail -20
```

## Change 2 — Fix production code (GREEN)

In `flush.py`, find `_write_ltm_graph` (or wherever the exception is caught).

The fix: in the `except` block, append the error string to the errors list (or re-raise).
Look at how other methods in `flush.py` handle errors — most append to an `errors` list
that is passed down or collected into `FlushResult`. Follow the same pattern.

Example fix (adjust to match actual method signature):
```python
except Exception as exc:
    log.warning("write_ltm_graph %s: %s", session_id, exc)
    errors.append(f"write_ltm_graph: {exc}")   # ← ADD THIS LINE
```

If `_write_ltm_graph` does not accept an `errors` parameter, add it:
```python
async def _write_ltm_graph(
    self,
    session_id: str,
    stm_data: dict,
    errors: list[str],   # add this
) -> dict[str, int]:
```
And update all call sites to pass the `errors` list.

## Acceptance Criteria
- `TestFlushSurfacesNeo4jError::test_neo4j_write_error_appears_in_flush_result_errors` passes (GREEN)
- Full unit suite still passes: `uv run pytest tests/ -q -m 'not integration' --tb=no 2>&1 | tail -5`
- `uv run ruff check --config ruff.toml graphknows/` — clean
- `uv run ruff format --check --config ruff.toml graphknows/` — clean

## Files to Modify
- `tests/test_flush.py` — add `TestFlushSurfacesNeo4jError`
- `graphknows/memory/flush.py` — `_write_ltm_graph` + caller: propagate error

## What NOT to do
- Do NOT change FlushResult schema (adding fields to it is ok if needed)
- Do NOT suppress or soften any other error handling
- Do NOT touch any other files
- Do NOT add noqa comments
- Do NOT commit
