# DEAD-001 — Remove Dead Code (Phase A)

## Context
Working directory: `/root/graphknows`. Confirm this before doing anything.

GraphKnows has been refactored. The following files are completely superseded and have
no active callers in the live production import graph. They must be deleted.

## Files to Delete

### Must delete together (they form a circular dependency cluster):
1. `graphknows/api.py` — 678-line FastAPI HTTP server, superseded by MCP stdio server.
   **Special:** `api.py` imports from `mcp/server.py`, and `mcp/server.py` has a comment
   saying it's "imported by api.py". Both must go together.
2. `graphknows/run.py` — uvicorn entry-point for api.py. 
3. `graphknows/mcp/server.py` — old 662-line HTTP FastMCP server, superseded by `stdio_server.py`.

### Delete independently:
4. `graphknows/graphgen/api.py` — 232-line REST wrapper for KnowledgePipeline. Self-contained.
5. `graphknows/graphgen/evaluation/lda_baseline.py` — LDA topic model baseline.
   **Special:** `graphknows/graphgen/evaluation/__init__.py` imports from it:
   ```python
   from .lda_baseline import (...)
   evaluate_lda_baseline,
   ```
   After deleting `lda_baseline.py`, you must also update `evaluation/__init__.py`
   to remove those imports and the `__all__` entries for them.

### Do NOT delete (optional/lazy import — safe to leave):
- `graphknows/graphgen/pipeline/visualization/thesis_plots.py` — it is only imported lazily
  inside `iterative_orchestrator.py:_generate_thesis_plots()` behind a try/except ImportError.
  It is in the "optional-path fragments" skip list in pyproject. Leave it.

## Additional cleanup: container.py
`graphknows/container.py` has a method `as_mcp_server()` that imports from `mcp/server.py`:
```python
from graphknows.mcp.server import mcp, set_dependencies
```
After deleting `mcp/server.py`, this import will break. You must update `container.py`:
- Remove the `as_mcp_server()` method entirely (it was a bridge to the old server).
- Also remove the docstring references to it in `__init__.py` and `container.py` if present.

## Test File to Delete
`tests/test_hybridrag_api.py` — imports `from graphknows.api import app`. Delete it.

## What NOT to touch
- `graphknows/mcp/stdio_server.py` — the LIVE server. Do not touch.
- `graphknows/mcp_ltm.py` — backward-compat shim importing from stdio_server. Leave.
- `graphknows/mcp/__init__.py` — already has the right comment. Leave.
- Any test file other than `test_hybridrag_api.py`.
- `graphknows/graphgen/pipeline/visualization/thesis_plots.py` — leave.
- `graphknows/graphgen/evaluation/lda_baseline.py` — wait, actually: SEE BELOW.

## Correction on lda_baseline.py
On second check: `lda_baseline.py` uses heavy optional deps (sklearn, pandas) that are
in the "optional-path fragments" skip list. The problem is only the re-export in `__init__.py`.
**Decision:** Delete `lda_baseline.py` AND clean up `evaluation/__init__.py` — remove
the import lines for `lda_baseline` from it.

## Acceptance Criteria
1. All listed files deleted from the repo.
2. `evaluation/__init__.py` no longer references `lda_baseline`.
3. `container.py` no longer references `mcp/server.py`.
4. `tests/test_hybridrag_api.py` deleted.
5. `uv run pytest tests/ -x -q -m "not integration"` passes (no new failures).
6. `ruff format graphknows/ && ruff check --no-fix graphknows/` — zero new errors.
7. `python3 -B -m py_compile graphknows/container.py graphknows/mcp/stdio_server.py graphknows/mcp_ltm.py` — all compile clean.

## Coding Standards
Read before starting:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

## Scope
Only touch the files listed above. Do not refactor, reformat, or "improve" other files.

## End-of-task summary required
```
Files deleted: <list>
Files modified: <list with one-line description each>
Files NOT touched: <confirm>
Follow-up needed: <list or "none">
```
