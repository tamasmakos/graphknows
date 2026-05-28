# Brief: Bug 1 — corpus_ingest asyncpg "attached to a different loop"

## Branch
`fix/mcp-bugs-round2` (already exists on origin). First step: `git checkout fix/mcp-bugs-round2`.

## Root Cause (confirmed)
`build_corpus_workflow()` in `graphknows/builders/stores.py` calls `_run(pg.connect())` at line ~105.
`_run` (from `graphknows/memory/ingestor.py`) submits to `_BG_LOOP` — a daemon thread's private event loop.
The asyncpg pool is therefore bound to `_BG_LOOP`.

`CorpusIngestWorkflow.run()` is async and runs on the MCP server's anyio event loop.
When it eventually calls asyncpg operations (via `flusher.flush()`), asyncpg detects the pool was created on
a different loop and raises "attached to a different loop" / "cannot perform operation: another operation is in progress".

`build_memory_ingestor()` also calls `_run(pg.connect())` at line ~79 — DO NOT TOUCH THAT FUNCTION.
It is self-consistent with `_BG_LOOP`. Only fix `build_corpus_workflow`.

## Fix — Two Files

### File 1: `graphknows/builders/stores.py`
In `build_corpus_workflow()` (starts around line 96):
- Remove the `from graphknows.memory.ingestor import _run` import (if it is only used for this connect call)
- Delete the `_run(pg.connect())` line
- Leave everything else unchanged

The resulting function builds and returns the workflow without connecting pg.

### File 2: `graphknows/memory/workflows/corpus_ingest.py`
At the top of `run()` (async method, line ~65), add a connect guard before any store usage:

```python
async def run(self, request: CorpusIngestRequest) -> CorpusIngestResult:
    # Ensure asyncpg pool is created on THIS event loop (not _BG_LOOP)
    if not getattr(self._pg, '_pool', None):
        await self._pg.connect()
    # ... rest of method unchanged ...
```

`self._pg` is a `PgVectorStore` — its pool attribute is `_pool`, set to `None` until `connect()` is awaited.
This guard prevents double-connect if `run()` is called more than once on the same workflow instance.

## Test Changes
File: `tests/builders/test_corpus_workflow_pg_connect.py` (already exists — check it first).

The test should verify:
1. `build_corpus_workflow()` does NOT call `pg.connect()` 
2. `workflow.run()` DOES call `pg.connect()` (mock it as an AsyncMock)

Update or extend the existing test file to cover these two cases. Do not add a third file.

## Acceptance Criteria
```bash
cd /root/graphknows
git checkout fix/mcp-bugs-round2
# make the changes
uv run pytest tests/builders/test_corpus_workflow_pg_connect.py -v
uv run pytest tests/ -q --tb=short 2>&1 | tail -10
uv run ruff check --config ruff.toml graphknows/ builders/ tests/
uv run ruff format --check --config ruff.toml graphknows/ tests/
uv run mypy graphknows/ --strict --exclude graphgen --exclude cli/memory --exclude feature_engineering --exclude forget --exclude mcp_ltm --exclude recall --exclude topic_summarizer --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores 2>&1 | tail -10
git add -A
git commit -m "fix: corpus_ingest pg.connect on correct event loop"
git push origin fix/mcp-bugs-round2
```

All tests must pass. No new mypy errors. Both ruff gates must pass.

## What NOT to do
- Do NOT touch `build_memory_ingestor()` — it is self-consistent with `_BG_LOOP`
- Do NOT modify `graphknows/memory/ingestor.py`
- Do NOT touch any other workflow files
- Do NOT open a GitHub PR
- Do NOT push to main
