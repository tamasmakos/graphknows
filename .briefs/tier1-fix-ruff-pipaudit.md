# Brief: Fix all ruff violations + pip-audit CI bug

## Working directory
/root/graphknows

## Goal
Fix ALL 51 ruff violations and fix the pip-audit CI job. No softening, no `noqa` additions.
Commit and push when done.

---

## Fix 1 — pip-audit CI job: add `--no-editable` to uv export

File: `.github/workflows/ci.yml`

Find the pip-audit step:
```yaml
        run: pip-audit --requirement <(uv export --no-dev --format requirements-txt) --progress-spinner off
```

Replace with:
```yaml
        run: pip-audit --requirement <(uv export --no-dev --no-editable --format requirements-txt) --progress-spinner off
```

This excludes the local editable `graphknows` package from the audit (it has no CVEs — it IS our package).

---

## Fix 2 — Auto-fix most ruff violations

Run the safe auto-fixer first:
```bash
cd /root/graphknows && uv run ruff check --fix --config ruff.toml graphknows/
```

This will auto-fix:
- RUF022 (`__all__` not sorted) — 4 files
- RUF100 (stale `noqa` directives) — many files
- RUF102 (invalid rule code in noqa)
- SIM103 (return condition directly)
- B007 (rename `j` to `_j`)

After running, check what remains:
```bash
uv run ruff check --config ruff.toml graphknows/ 2>&1
```

---

## Fix 3 — Remaining violations requiring manual edits

### 3a. TRY401 — redundant exception in logging.exception
File: `graphknows/container.py:190`

Find something like:
```python
logging.exception("...", exc)
# or
logger.exception("...", exc)
```
The exception object is implicit in `logger.exception` — remove the explicit arg.
Change `logger.exception("message", exc)` → `logger.exception("message")`

### 3b. B905 — zip() without strict= (5 locations)
For every `zip(a, b)` without `strict=`, add `strict=False` to be explicit:
- `graphknows/container.py:265`
- `graphknows/ontology/router.py:101`
- `graphknows/ontology/router.py:131`
- `graphknows/retrieval/retriever.py:123`
- `graphknows/retrieval/retriever.py:448`
- `graphknows/retrieval/stores/neo4j.py:100`

Pattern: `zip(a, b)` → `zip(a, b, strict=False)`

### 3c. RUF006 — store reference to asyncio.create_task
File: `graphknows/container.py:308`

Find `asyncio.create_task(...)` without storing the result.
Add `_task = ` prefix to store the reference, then add a docstring comment nearby explaining it's tracked via the container's task registry, OR add to a task list.

If the container already has a `_tasks` set or list, do:
```python
self._background_tasks.add(asyncio.create_task(...))
```

If no such attribute exists, just store in a throwaway local:
```python
_bg_task = asyncio.create_task(...)  # tracked by event loop until done
```

### 3d. SIM105 — use contextlib.suppress
Two locations:
- `graphknows/feature_engineering.py:555`
- `graphknows/forget.py:66`

Pattern — replace:
```python
try:
    <single statement>
except SomeException:
    pass
```
With:
```python
import contextlib
with contextlib.suppress(SomeException):
    <single statement>
```
Make sure `import contextlib` is at the top of the file (it's stdlib — always available).

### 3e. SIM115 — use context manager for opening files
File: `graphknows/memory/ingestor.py:135`

Find bare `open(...)` without `with`. Wrap it:
```python
# Before
f = open(path, ...)  # noqa: WPS515
...use f...

# After
with open(path, ...) as f:
    ...use f...
```
Remove the `# noqa: WPS515` comment — that rule is no longer needed.

### 3f. TRY300 — move return to else block (3 locations)
Files:
- `graphknows/mcp/stdio_server.py:742`
- `graphknows/memory/flush.py:146`
- `graphknows/ontology/rel_extractor.py:137`

Pattern:
```python
# Before
try:
    result = something()
    return result  # ← this is inside try
except SomeError:
    handle()

# After
try:
    result = something()
except SomeError:
    handle()
else:
    return result  # ← moved to else block
```

### 3g. RUF002/003 — EN dash in docstrings/comments
Replace every `–` (EN DASH, U+2013) with `-` (HYPHEN-MINUS) in:
- `graphknows/feature_engineering.py:376, 535, 541`
- `graphknows/mcp/stdio_server.py:474, 598, 638, 676`
- `graphknows/ontology/router.py:87`
- `graphknows/recall.py:1, 64, 124, 193`
- `graphknows/retrieval/context.py:67`
- `graphknows/retrieval/stores/neo4j.py:599`

Simple sed replacement: `–` → `-`

---

## Verification

After all fixes, run:
```bash
cd /root/graphknows && uv run ruff check --config ruff.toml graphknows/ 2>&1
```
Expected: `All checks passed!` (0 errors)

Then run tests to confirm nothing broke:
```bash
cd /root/graphknows && uv run pytest tests/ -q -m 'not integration' --tb=short 2>&1 | tail -20
```
Expected: all pass (currently 232 passed).

---

## Acceptance criteria

1. `uv run ruff check --config ruff.toml graphknows/` → `All checks passed!`
2. Tests: 232+ passed, 0 failed
3. `git diff --name-only` shows ONLY files in graphknows/ + .github/workflows/ci.yml
4. Commit with message:
   ```
   fix: resolve all 51 ruff violations and fix pip-audit editable install error

   - ci.yml: add --no-editable to uv export in pip-audit job
   - ruff auto-fix: RUF022 __all__ sorting, RUF100/102 stale noqa, SIM103, B007
   - manual: TRY401 remove redundant exc arg from logger.exception
   - manual: B905 add strict=False to all zip() calls
   - manual: RUF006 store asyncio.create_task reference
   - manual: SIM105 replace try/except/pass with contextlib.suppress
   - manual: SIM115 wrap bare open() in context manager
   - manual: TRY300 move return to else blocks (3 sites)
   - manual: RUF002/003 replace EN dashes with hyphens in docstrings/comments
   ```
5. Push to main.

## What NOT to do
- Do NOT add `noqa` to suppress any violation
- Do NOT modify test files
- Do NOT touch graphgen/ production code
- Do NOT change any logic — all fixes are mechanical style/safety fixes
