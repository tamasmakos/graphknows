# GDS-001 — Consolidate GDS Enrichment Paths + Expose run_gds in MCP flush

## Context
Working directory: `/root/graphknows`. Confirm this before doing anything.

There are currently TWO overlapping GDS paths in the memory layer:

**Path A (lightweight) — `graphknows/retrieval/stores/neo4j.py::run_gds_enrichment()`**
- Runs: PageRank + betweenness only
- Called from: `memory/flush.py::_run_gds()` when `run_gds=True`
- Currently hardcoded to `run_gds=False` in the MCP `memory_flush` tool

**Path B (full) — `graphknows/feature_engineering.py::run_feature_engineering()`**
- Runs: Node2Vec → cosine edge weights → re-project → Leiden/Louvain → PageRank + betweenness + degree
- Called from: `graphknows/cli/memory.py::flush_cmd()` via `LTMFeatureEngineer.run()`
- NOT called from MCP or any real-time path

The goal of this task is:
1. Make `run_gds` a configurable parameter on the MCP `memory_flush` tool (currently hardcoded False)
2. Wire the full `feature_engineering.py` path as the canonical enrichment when `run_gds=True`
   (so MCP flush can trigger Node2Vec+Louvain+PageRank, not just the lightweight path)
3. Keep `neo4j.py::run_gds_enrichment()` as the thin wrapper it is — but have it call
   `run_feature_engineering()` from `feature_engineering.py` instead of duplicating GDS logic

## Files to Modify

### 1. `graphknows/mcp/stdio_server.py` — expose `run_gds` on the `memory_flush` tool
Currently (around line 435):
```python
run_gds=False,
```
Change the `memory_flush` tool to accept a `run_gds: bool = False` parameter and pass it through.

### 2. `graphknows/retrieval/stores/neo4j.py` — wire `run_gds_enrichment` to feature_engineering
Currently `run_gds_enrichment(session_id)` has its own GDS queries (PageRank + betweenness).
Replace the body with a call to `run_feature_engineering(self._driver)` from `feature_engineering.py`:
```python
async def run_gds_enrichment(self, session_id: str) -> dict[str, Any]:
    from graphknows.feature_engineering import run_feature_engineering
    return await asyncio.to_thread(run_feature_engineering, self._driver)
```
This makes the lightweight path canonical to the full path.

### 3. `tests/test_mcp_stdio_server.py` — update any test that checks `memory_flush` signature
If there's a test that calls `memory_flush` and checks its parameters, update it to pass
`run_gds=False` explicitly (the default) so it still works.

## What NOT to touch
- `feature_engineering.py` itself — do not modify it
- `memory/flush.py` — do not modify (it already has `run_gds` in `FlushOptions`)
- `cli/memory.py` — do not modify
- Any graphgen files

## Acceptance Criteria
1. `memory_flush` MCP tool accepts `run_gds: bool = False` and passes it to `SessionFlusher.flush()`.
2. `run_gds_enrichment` in `neo4j.py` delegates to `run_feature_engineering` (no duplicate GDS code).
3. `uv run pytest tests/ -x -q -m "not integration"` passes — no new failures.
4. `ruff format graphknows/ && ruff check --no-fix graphknows/` — zero new errors.
5. `python3 -B -m py_compile graphknows/mcp/stdio_server.py graphknows/retrieval/stores/neo4j.py` — clean.

## Coding Standards
Read before starting:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

## Scope
Only modify the three files listed above. Do not refactor, rename, or improve anything else.

## End-of-task summary required
```
Files changed: <list with one-line description>
Files NOT touched: <confirm>
Follow-up needed: <list or "none">
```
