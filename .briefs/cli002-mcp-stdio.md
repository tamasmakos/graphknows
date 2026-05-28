# Brief: CLI-002 — Standalone MCP stdio server

## Context
Project root: `/root/graphknows`
Package: `graphknows/` (single package)
Target file: `graphknows/graphknows/mcp/stdio_server.py` (NEW)
Existing MCP package: `graphknows/graphknows/mcp/` — has `__init__.py` (empty, do not touch) and `server.py` (FastAPI-coupled, do not touch)
pyproject.toml: `/root/graphknows/pyproject.toml`
Venv: `/root/graphknows/.venv`
Run tests: `cd /root/graphknows && uv run pytest tests/ -x -q -m "not integration"`

Read BEFORE writing any code:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

Check `mcp` package availability:
```bash
cd /root/graphknows && /root/graphknows/.venv/bin/python -c "from mcp.server.fastmcp import FastMCP; print('ok')"
```
If that fails, check what MCP API is available:
```bash
/root/graphknows/.venv/bin/python -c "import mcp; print(dir(mcp))"
/root/graphknows/.venv/bin/python -c "import mcp.server; print(dir(mcp.server))"
```
Adapt the implementation to whatever API is actually installed. Do NOT assume FastMCP exists — verify first.

## Task
Build a standalone MCP stdio server at `graphknows/graphknows/mcp/stdio_server.py`.

MCP stdio = JSON-RPC 2.0 over stdin/stdout. The process reads requests from stdin, writes responses to stdout. This makes graphknows discoverable by ANY MCP client (Claude Desktop, Cursor, Zed, Hermes native MCP, custom agents) without needing the full FastAPI service.

## Tools to expose

### 1. `memory_ingest`
- Description: "Ingest text into graphknows short-term memory (Memgraph + pgvector STM)"
- Input schema: `{text: string, session_id: string, title?: string}`
- Returns: `{"file_id": str, "chunks": int, "entities": int, "elapsed_s": float, "errors": []}`

### 2. `memory_recall`
- Description: "Recall from graphknows memory using hybrid RRF search"
- Input schema: `{query: string, session_id: string, top_k?: integer (default 5), scope?: string (default "stm", also "ltm", "both")}`
- Returns: `{"hits": [{"score": float, "type": str, "text": str, "session_id": str}]}`

### 3. `memory_stats`
- Description: "Get graphknows memory statistics (STM and LTM node counts)"
- Input schema: `{session_id?: string}`
- Returns: `{"stm": {"sessions": int, "nodes": int, "edges": int}, "ltm": {"entities": int, "topics": int, "chunks": int}}`

### 4. `memory_flush`
- Description: "Flush a session from short-term memory to long-term memory (Memgraph → Neo4j)"
- Input schema: `{session_id: string}`
- Returns: `{"nodes": int, "edges": int, "chunks": int, "turns": int, "elapsed_s": float, "errors": []}`

### 5. `memory_doctor`
- Description: "Check graphknows backend connectivity (Memgraph, Neo4j, pgvector)"
- Input schema: `{}` (no inputs)
- Returns: `{"memgraph": bool, "neo4j": bool, "pgvector": bool}`

## Implementation approach

### Option A — FastMCP (preferred if available)
```python
from mcp.server.fastmcp import FastMCP

app = FastMCP("graphknows-memory")

@app.tool()
async def memory_doctor() -> dict:
    \"\"\"Check graphknows backend connectivity.\"\"\"
    ...

def main() -> None:
    import asyncio
    asyncio.run(app.run_stdio_async())
```

### Option B — Low-level MCP server (if FastMCP unavailable)
Use `mcp.server.Server` + `mcp.server.stdio.stdio_server` directly. Define tools via `@server.list_tools()` and `@server.call_tool()` handlers.

Check which is available and use accordingly. The brief describes the desired *behaviour*, not the exact API — adapt.

## Backend wiring
Reuse helper functions from `graphknows/cli/memory.py`:
```python
from graphknows.cli.memory import _mg_uri, _pg_dsn, _embed_fn
```

Lazy singleton stores (init on first call, not at import):
```python
_mg: Any = None
_pg: Any = None

def _get_stores():
    global _mg, _pg
    if _mg is None:
        from graphknows.retrieval.stores.memgraph import MemgraphStore
        from graphknows.retrieval.stores.pgvector import PgVectorStore
        from graphknows.memory.ingestor import _run
        _mg = MemgraphStore(_mg_uri())
        _pg = PgVectorStore(_pg_dsn())
        _run(_pg.connect())
    return _mg, _pg
```

GLiNER2 stays lazy — only loaded when `memory_ingest` is actually called.

## Entry point in pyproject.toml
Find the `[project.scripts]` section in `/root/graphknows/pyproject.toml` and add:
```toml
graphknows-mcp = "graphknows.mcp.stdio_server:main"
```
Then reinstall: `cd /root/graphknows && uv pip install -e . --quiet`
Verify: `which graphknows-mcp` should point to `/root/graphknows/.venv/bin/graphknows-mcp`

## Critical constraints
- No FastAPI, no uvicorn, no HTTP — pure stdio only
- Importable without loading GLiNER2: `python3 -c "from graphknows.mcp.stdio_server import main"` must not trigger model load
- Load `.env` from PWD at module level: `from dotenv import load_dotenv; load_dotenv()`
- All logging to stderr; MCP protocol (JSON-RPC) to stdout
- Do NOT modify `graphknows/mcp/__init__.py` — the circular import note must be preserved
- Do NOT modify `graphknows/mcp/server.py` — that is the FastAPI-coupled server, out of scope

## Tests
`tests/test_mcp_stdio_server.py` — unit tests with mocked stores:

```python
def test_tools_registered():
    \"\"\"All 5 tools must be registered on the MCP app.\"\"\"
    from graphknows.mcp.stdio_server import app
    # Verify tool names present — exact API depends on FastMCP vs low-level
    ...

def test_memory_doctor_returns_dict():
    \"\"\"memory_doctor returns expected keys with bool values.\"\"\"
    # mock MemgraphStore, PgVectorStore, Neo4jStore connectivity
    ...
```

Tests must not require live infra. Mock at the store level. Mark integration tests with `@pytest.mark.integration`.

## Acceptance criteria
1. `python3 -c "from graphknows.mcp.stdio_server import main"` — no error, no model load
2. `which graphknows-mcp` → `/root/graphknows/.venv/bin/graphknows-mcp`
3. Tools/list probe:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | graphknows-mcp 2>/dev/null
   ```
   → JSON-RPC response with all 5 tool names in `result.tools[].name`
4. Doctor probe:
   ```bash
   echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_doctor","arguments":{}}}' | graphknows-mcp 2>/dev/null
   ```
   → response with `{"memgraph":true,"neo4j":true,"pgvector":true}` (or appropriate values)
5. `uv run pytest tests/ -x -q -m "not integration"` passes

## Do NOT touch
- `graphknows/mcp/__init__.py`
- `graphknows/mcp/server.py`
- `graphknows/cli/memory.py` (read only, do not modify)
- Any file outside the new `stdio_server.py`, `pyproject.toml`, and test file

## End-of-task summary required
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Follow-up needed: <list or none>
```
