# STM-009: CI gate — run full test suite, verify green build

## Context
Project root: /root/graphknows
Model: github-copilot/claude-sonnet-4.6

## Goal
Run the full CI gate and get to 100% green. Fix any import errors, missing deps, or test failures.
This is the final integration check — all STM-001..STM-008 code must pass CI here.

## What CI runs (from services/graphknows/)
```bash
cd /root/graphknows/services/graphknows
set -a && source ../../.env && set +a

# 1. Import check
python3 scripts/check_imports.py

# 2. Lint
ruff format --check . --config ruff.toml
ruff check --no-fix --config ruff.toml .

# 3. Type check
python3 -m mypy graphknows/ --config-file pyproject.toml --ignore-missing-imports

# 4. Tests (unit only — no integration)
python3 -B -m pytest tests/ -x -q -m "not integration"
```

## Your job
1. Run the full CI gate.
2. For any failure: read the error, identify the root cause, fix it minimally.
   - Import error? Check pyproject.toml deps and check_imports.py.
   - Ruff error? Fix the specific line — do NOT reformat unrelated files.
   - mypy error? Fix the type annotation — do NOT add `type: ignore` unless truly needed.
   - Test failure? Read the test, fix the production code (not the test) unless the test is wrong.
3. Re-run CI gate after each fix.
4. Keep fixing until the gate is 100% green.
5. Do NOT add # noqa or # type: ignore to suppress real errors — fix the root cause.

## Common pitfalls to watch for
- Missing `python-docx` in pyproject.toml → add it
- asyncio.get_event_loop() deprecation in Python 3.10+ → use asyncio.new_event_loop() or run_until_complete pattern
- `from __future__ import annotations` needed for `X | Y` type hints in 3.9
- MemgraphStore methods added in STM-001 might have wrong param names vs call sites in STM-003/005
- PgVectorStore async methods: ensure they're truly async (async def + await)
- `recall.py` and `document_ingestor.py` must be git rm'd before CI — check with `git status`
- topic_summarizer.py must be git rm'd
- Any import of deleted modules will cause ImportError

## After green CI
Run: `git diff --name-only HEAD` and report all changed files.
Run: `python3 -B -m pytest tests/ -v -m "not integration" 2>&1 | tail -30`

## Acceptance criteria
- CI gate exits 0 across all 4 steps
- Test count >= 35 (covers parsers, ingestor, dialog, flush, recall_stm, recall_ltm, all CLI commands)
- Zero ruff errors, zero mypy errors
- check_imports.py passes (all declared deps importable)
- git status shows no unexpected untracked files or deleted files
