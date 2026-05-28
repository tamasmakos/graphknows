# RECALL-001: Strip ## Content prefix before embedding and storage

## Working directory
`/root/graphknows`

## Coding standards (read these first)
- `.hermes/instructions/python-style.instructions.md`
- `.hermes/instructions/tdd-test-driven-development.instructions.md`

## Problem
`MemoryIngestor._ingest_parsed()` in `graphknows/memory/ingestor.py` embeds `seg.text`
verbatim. The plaintext formatter wraps chunks in `## Content\n\n` before writing to
`seg.text`, so the stored text and its embedding are both polluted.

Confirmed live: all 4 rows in `ltm_chunks` start with `## Content\n\n`.

The `_clean()` strip in `mcp/stdio_server.py:315` strips it at display time only —
the vector was already computed on dirty text. **Do not touch stdio_server.py.**

## Root cause
`graphknows/memory/ingestor.py` — specifically `_ingest_parsed()`:
- `embedding = self._embed_fn(seg.text)` — embeds dirty text
- `text=seg.text` in write_chunk / write_stm_chunk calls — stores dirty text

## Fix
Add a module-level compiled regex and a `_strip_content_prefix()` helper.
Apply it once to produce `clean_text`, use `clean_text` everywhere `seg.text` was used
for embedding and storage. Keep `seg.text` unchanged (don't mutate the segment).

```python
import re

_CONTENT_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^##\s*Content\s*\n+", re.IGNORECASE
)

def _strip_content_prefix(text: str) -> str:
    """Strip '## Content\\n' heading artifact from chunk text before embedding."""
    return _CONTENT_PREFIX_RE.sub("", text).strip()
```

In `_ingest_parsed()`:
```python
clean_text = _strip_content_prefix(seg.text)
embedding: list[float] = self._embed_fn(clean_text)
# use clean_text instead of seg.text in all write_chunk / write_stm_chunk calls
```

## Files to touch (ONLY these)
1. `graphknows/memory/ingestor.py` — add helper, use clean_text for embed + store
2. `tests/test_memory_ingestor.py` — add test asserting stored text has no `## Content` prefix

## Acceptance criteria
- New unit test: construct a segment with text `"## Content\n\nActual text"`, ingest it,
  assert the stored chunk text captured by mock equals `"Actual text"`
- All existing tests still pass
- `ruff format graphknows/ && ruff check --no-fix --config ruff.toml graphknows/` clean
- `mypy graphknows/memory/ingestor.py` clean

## What NOT to touch
- `mcp/stdio_server.py` — leave `_clean()` there, it stays as belt-and-suspenders
- Any other file not listed above
- No refactors, no renames, no unrelated improvements

## End-of-task summary format (mandatory)
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Tests added: <list>
Test results: <pytest output tail>
```
