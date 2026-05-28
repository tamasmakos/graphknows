# MEM-005: Add version/pid to memory_doctor output

## Context
- Kanban: t_267d8737
- Working dir: /root/graphknows
- Venv: /root/graphknows/.venv
- Model: openrouter/deepseek/deepseek-v4-flash

## Problem
The MCP server is a long-lived subprocess launched by Hermes. After code changes,
the old process keeps running old code. There's no way to verify the running process
is current without checking the return of memory_doctor.

## Your task

### 1. Edit graphknows/mcp/stdio_server.py — memory_doctor() function

Add two new keys to the returned dict:
- `"version"`: git short hash of HEAD. Use:
  ```python
  import subprocess
  try:
      v = subprocess.check_output(
          ["git", "rev-parse", "--short", "HEAD"],
          cwd="/root/graphknows", text=True, timeout=3
      ).strip()
  except Exception:
      v = "unknown"
  ```
- `"pid"`: `os.getpid()`

Return dict should be: `{"memgraph": bool, "neo4j": bool, "pgvector": bool, "version": str, "pid": int}`

### 2. Edit tests/test_mcp_stdio_server.py

Add a test asserting:
- `memory_doctor` result contains `"version"` key (string, non-empty)
- `memory_doctor` result contains `"pid"` key (positive integer)

The existing test for doctor checks `memgraph/neo4j/pgvector` keys — keep it, add to it or add a separate test.

## Acceptance criteria
- memory_doctor returns version and pid
- Test passes: `cd /root/graphknows && .venv/bin/python -m pytest tests/test_mcp_stdio_server.py -v`
- All existing tests still pass
- Do not commit
