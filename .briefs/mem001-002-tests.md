# MEM-001+002: Tests for LTM session_filter fix and Content prefix strip

## Context
- Kanban: t_f13bf42b
- Working dir: /root/graphknows
- Venv: /root/graphknows/.venv
- Model: openrouter/deepseek/deepseek-v4-flash

## What was already fixed (do NOT change this)
File: graphknows/mcp/stdio_server.py

Fix 1 (line ~335): LTM recall used `session_filter=session_id` which restricts search
to the current session only. Changed to `session_filter=None` so LTM searches all sessions.

Fix 2 (line ~300): recall results had a `## Content\n` heading artifact. Added _clean()
with re.sub to strip it.

## Your task
Add tests to: tests/test_mcp_stdio_server.py

### Test 1 — LTM uses session_filter=None
The `memory_recall` function calls `recall_long_term(..., session_filter=...)`.
When scope="ltm", it must pass `session_filter=None`.

Pattern: mock `recall_long_term` and assert it was called with `session_filter=None`
regardless of what `session_id` was passed to `memory_recall`.

### Test 2 — Content prefix is stripped
The `_clean()` helper inside `_fmt()` strips the `## Content\n` prefix.
Test it directly by patching or extracting the logic:
- Input: `"## Content\n\nSome real content here."`
- Expected output: `"Some real content here."`
- Also test: `"## content\n\nLower case."` → `"Lower case."` (case-insensitive)
- Also test: text without the prefix is returned unchanged

### Test 3 — ensure existing 6 tests still pass
Run: `cd /root/graphknows && .venv/bin/python -m pytest tests/test_mcp_stdio_server.py -v`
All 6 must pass.

## Acceptance criteria
- 2+ new tests added to test_mcp_stdio_server.py
- All existing + new tests pass
- No production code changed
- Do not commit
