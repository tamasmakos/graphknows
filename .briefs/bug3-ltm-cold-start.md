# Brief: Bug 3 — ltm_query cold-start timeout

## Branch
`fix/mcp-bugs-round2` (already exists on origin). First step: `git checkout fix/mcp-bugs-round2`.

## Root Cause (confirmed)
This codebase uses the OpenRouter HTTP API for embeddings (NOT a local sentence-transformer model).
`embed_one()` in `graphknows/retrieval/stores/embedder.py` makes an HTTP call to OpenRouter on every invocation.

Cold-start latency comes from:
1. asyncpg connection setup on first call (pool not yet connected)
2. First OpenRouter HTTP call (new HTTP connection setup)
3. httpx timeout is 30s — if OpenRouter is slow, this eats most of the MCP 60s budget

**Fix direction:** Pre-warm both connections at MCP server startup so the first tool call never pays cold-start cost.

## Fix — Two Files

### File 1: `graphknows/mcp/stdio_server.py`
Current `main()` function calls `asyncio.run(app.run_stdio_async())`.

Add a warmup coroutine before the server starts:

```python
async def _startup_warmup() -> None:
    """Pre-warm embedder and pgvector pool before first tool call."""
    import logging
    log = logging.getLogger("graphknows.startup")

    # 1. Probe embedder (validates API key + network reachability)
    try:
        from graphknows.retrieval.stores.embedder import embed_one
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, embed_one, "warmup")
        log.warning("Embedder warmup OK")
    except Exception as exc:
        log.warning("Embedder warmup failed (non-fatal): %s", exc)

    # 2. Pre-connect the LTM pgvector store (if accessible via state module)
    try:
        from graphknows.mcp._state import get_settings  # or whatever the state accessor is
        # Just importing settings is enough to validate config
        get_settings()
        log.warning("Settings warmup OK")
    except Exception as exc:
        log.warning("Settings warmup failed (non-fatal): %s", exc)


async def _run_server() -> None:
    await _startup_warmup()
    await app.run_stdio_async()


def main() -> None:
    """Run the graphknows MCP stdio server."""
    asyncio.run(_run_server())
```

**Important:** Read `graphknows/mcp/stdio_server.py` and `graphknows/mcp/_state.py` first to understand the exact import paths available. Adapt the warmup to use whatever state accessors exist. The warmup must be non-fatal (try/except everything).

### File 2: `graphknows/retrieval/stores/embedder.py`
Change the httpx timeout from a flat 30s integer to a structured timeout object so slow OpenRouter calls
fail with a clear error rather than silently consuming the MCP budget:

```python
# BEFORE (line ~60):
timeout=30,

# AFTER:
timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
```

This requires `httpx.Timeout` — already available since `httpx` is already imported.

## Test Changes
File: `tests/mcp_server/` — add a new test file `test_startup_warmup.py` (if it doesn't exist):

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

async def test_startup_warmup_calls_embed_one():
    """Warmup should invoke embed_one once."""
    from graphknows.mcp.stdio_server import _startup_warmup
    with patch("graphknows.retrieval.stores.embedder.embed_one") as mock_embed:
        mock_embed.return_value = [0.1, 0.2]
        await _startup_warmup()
    mock_embed.assert_called_once()

async def test_startup_warmup_non_fatal():
    """Exceptions in warmup must not propagate."""
    from graphknows.mcp.stdio_server import _startup_warmup
    with patch("graphknows.retrieval.stores.embedder.embed_one", side_effect=RuntimeError("API down")):
        await _startup_warmup()  # must not raise
```

Adapt the test to the actual implementation — the key invariants are: embed_one is called, and exceptions don't propagate.

## Acceptance Criteria
```bash
cd /root/graphknows
git checkout fix/mcp-bugs-round2
# make the changes
uv run pytest tests/mcp_server/ -v
uv run pytest tests/ -q --tb=short 2>&1 | tail -10
uv run ruff check --config ruff.toml graphknows/ tests/
uv run ruff format --check --config ruff.toml graphknows/ tests/
uv run mypy graphknows/ --strict --exclude graphgen --exclude cli/memory --exclude feature_engineering --exclude forget --exclude mcp_ltm --exclude recall --exclude topic_summarizer --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores 2>&1 | tail -10
git add -A
git commit -m "fix: ltm_query cold-start — startup warmup + httpx structured timeout"
git push origin fix/mcp-bugs-round2
```

All tests must pass. No new mypy errors. Both ruff gates must pass.

## What NOT to do
- Do NOT add a local model loading step — there is no sentence-transformer in this codebase
- Do NOT modify `build_memory_ingestor` or `build_corpus_workflow`
- Do NOT touch any retrieval store logic beyond the timeout change in `embedder.py`
- Do NOT open a GitHub PR
- Do NOT push to main
