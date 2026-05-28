# UNIFY-001: Merge MCP Servers into Single Unified Server

## First actions (mandatory)
At the top of your final reply, explicitly state:
1. current working directory
2. what tools/capabilities you actually have in this session
3. whether you can read repo files and run local python

If cwd is not `/root/graphknows`, stop and fix it first.

## Hard scope
Modify only:
- `graphknows/mcp/stdio_server.py` — add 6 LTM tools from `graphknows/mcp_ltm.py`
- `graphknows/mcp_ltm.py` — convert to thin backward-compat re-export
- `tests/test_mcp_stdio_server.py` — add tests for new LTM tools

Do NOT modify any other production files.
Do NOT commit, push, or open a PR.

## Project rules
- Repo root: `/root/graphknows`
- Use local instructions: `.hermes/instructions/python-style.instructions.md`, `.hermes/instructions/tdd-test-driven-development.instructions.md`
- Test command: `cd /root/graphknows && uv run pytest tests/ -x -q -m "not integration" 2>&1 | tail -10`
- All 209 existing tests must still pass

## Context — what we're doing

GraphKnows has TWO MCP servers today:
1. `graphknows/mcp/stdio_server.py` — FastMCP app with 5 tools (memory_doctor, memory_stats, memory_ingest, memory_recall, memory_flush). Running as `graphknows-mcp` binary.
2. `graphknows/mcp_ltm.py` — lower-level mcp.server.Server with 6 tools (ltm_query, ltm_entity, ltm_topics, recall_by_document, recall_by_dialog, recall_by_topic). NOT wired to Hermes.

We are merging them into ONE unified FastMCP server in `stdio_server.py`. This gives any MCP-speaking agent (Hermes, OpenCode, Claude Desktop) all 11 memory tools from one connection.

The product pivot: GraphKnows is now a pure domain-free memory service. No EU connectors, no investigation agent, no report generation.

## Read before coding

Read these files in order:
1. `graphknows/mcp/stdio_server.py` — understand the FastMCP pattern, @app.tool() decorator, helper functions (_mg_uri, _pg_dsn, _neo4j_creds, _embed_fn, _get_mg, _get_pg, _get_neo)
2. `graphknows/mcp_ltm.py` — understand the 6 LTM tools, their dispatch pattern, and implementation details
3. `graphknows/mcp/__init__.py` — check for exports
4. `tests/test_mcp_stdio_server.py` — existing test patterns

## What to build

### Step 1: Add 6 LTM tools to stdio_server.py

Copy each tool from `mcp_ltm.py` and convert to FastMCP `@app.tool()` pattern.

**Conversion rules for each tool:**

For sync tools (ltm_entity, ltm_topics, recall_by_document, recall_by_dialog, recall_by_topic):
```python
@app.tool()
def ltm_entity(name: str) -> dict[str, Any]:
    """Look up a named entity in the LTM Neo4j graph..."""
    # copy implementation from _tool_ltm_entity
```

For async tools (ltm_query):
```python
@app.tool()
async def ltm_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """Semantic search over LTM topic summaries..."""
    # copy implementation from _tool_ltm_query, remove dispatch wrapper
```

**Key adaptations:**
- Use the existing helper functions from stdio_server.py (_neo4j_creds, etc.) — do NOT re-import neo4j driver separately
- `_tool_ltm_query` uses `asyncpg` — this is fine, keep the asyncpg import at function level
- `_tool_ltm_entity` uses sync Neo4j driver via `_neo4j_driver()` — replace with `_get_neo()` or `_neo4j_creds()`
- `recall_by_document`, `recall_by_dialog`, `recall_by_topic` import from `graphknows.recall` — these imports are already `from graphknows.recall import ...` in mcp_ltm.py, use same pattern
- Remove the `_dispatch` wrapper pattern from mcp_ltm.py — FastMCP handles dispatch via decorators
- All return types should be `dict[str, Any]` or strings (FastMCP auto-converts)

**Important: The `recall_by_*` tools in mcp_ltm.py return JSON strings (from `json.dumps(...)`). In FastMCP, prefer returning dicts — the framework handles serialization. Convert json.dumps returns to dict returns.**

### Step 2: Update app instructions

Add the new tools to the FastMCP app instructions string so the LLM knows about them:

```python
app = FastMCP(
    "graphknows-memory",
    instructions=(
        "graphknows-memory exposes short-term and long-term knowledge-graph "
        "backed memory. Use memory_ingest to store content, memory_recall to "
        "retrieve it, memory_flush to promote session memory to long-term, "
        "memory_doctor to verify backend connectivity, and memory_stats for counts. "
        "For LTM exploration: ltm_query searches topics semantically, "
        "ltm_entity looks up named entities, ltm_topics lists topics by threshold. "
        "recall_by_document, recall_by_dialog, and recall_by_topic provide "
        "entity-matched retrieval across documents, conversations, and topics."
    ),
)
```

### Step 3: Convert mcp_ltm.py to backward-compat re-export

Replace `mcp_ltm.py` content with a thin re-export that imports from `graphknows.mcp.stdio_server` so existing imports don't break:

```python
"""Backward-compat re-export. Use graphknows.mcp.stdio_server directly."""
from graphknows.mcp.stdio_server import (
    app,
    main,
    # export specific tool functions if needed for direct import
)
```

Keep the `asyncio.run(mcp_main())` entry point pattern if anything calls it directly.

### Step 4: Add tests

Add test functions in `tests/test_mcp_stdio_server.py`:

Test that the 6 new LTM tools are registered:
```python
def test_ltm_tools_registered():
    """All 6 LTM tools are registered in the FastMCP app."""
    from graphknows.mcp.stdio_server import app
    tool_names = {t.name for t in app._tool_manager._tools.values()}
    for name in ("ltm_query", "ltm_entity", "ltm_topics", 
                 "recall_by_document", "recall_by_dialog", "recall_by_topic"):
        assert name in tool_names, f"Missing tool: {name}"
```

Test ltm_topics with mocked Neo4j:
```python
def test_ltm_topics_returns_topics():
    """ltm_topics returns topic list above threshold."""
    # Mock neo4j session, verify return shape
```

Test ltm_entity with mocked Neo4j:
```python
def test_ltm_entity_not_found():
    """ltm_entity returns error dict when entity not found."""
```

Add at least 3 test functions. Follow the existing mock patterns in the file (mock neo4j/pgvector, use MonkeyPatch).

## Acceptance criteria

- [ ] All 11 tools registered in stdio_server.py FastMCP app
- [ ] mcp_ltm.py is a thin re-export (no duplicative logic)
- [ ] All 209+ existing tests pass
- [ ] New tests for LTM tools pass
- [ ] No production files outside stdio_server.py, mcp_ltm.py, and tests modified
- [ ] `graphknows/mcp/__init__.py` updated if needed

## Git — do NOT commit

Work in the working tree. Do not commit. The principal engineer will review and integrate.