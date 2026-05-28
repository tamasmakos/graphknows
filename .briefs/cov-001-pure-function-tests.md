# BRIEF: COV-001 — Pure-Function Coverage Tests

## Goal
Write meaningful unit tests that cover the untested pure-function code in:
1. `graphknows/recall.py` — currently 0% covered (59 lines)
2. `graphknows/retrieval/context.py` — currently 40% covered, hit lines 103+
3. `graphknows/graphgen/utils/utils.py` — 0% covered
4. `graphknows/graphgen/utils/provenance.py` — 0% covered  
5. `graphknows/graphgen/utils/diagnostics.py` — 0% covered
6. `graphknows/graphgen/utils/labels.py` — 0% covered
7. `graphknows/graphgen/utils/schema_utils.py` — 0% covered (skip — requires networkx+schema)
8. `graphknows/graphgen/utils/logging.py` — 0% covered
9. `graphknows/graphgen/pipeline/graph_cleaning/canonicalization.py` — 0% covered
10. `graphknows/graphgen/pipeline/graph_cleaning/pruning.py` — 0% covered

**Current overall coverage: 26%. Target: >80%.**

## Philosophy (mandatory — read before writing a single line)

**Meaningful tests fail when you break the code.** We do NOT want:
- Tests that assert `.run()` was called once
- Tests that mock out everything and test nothing
- Tests that assert "no exception raised" with trivial inputs
- Tests that always pass regardless of implementation

**We DO want:**
1. **Tests that exercise real behavior** — real inputs, real outputs, observable side effects
2. **Tests for error paths** — empty input, None, boundary values
3. **Tests that would fail if you change the logic** — flip a comparison, wrong regex, etc.
4. **Arrange-Act-Assert** pattern in every test

## Working directory
`/root/graphknows`

## Venv
`uv run pytest` — always use this, not `python -m pytest` directly.

## What to create

### File: `tests/test_recall_module.py` (NEW)
Test `graphknows/recall.py`:

```python
"""Tests for the recall module entity-match recall functions."""
# Key behaviors to test:
# 1. _extract_query_tokens: 
#    - Normal sentence → list of unique tokens ≥3 chars, lowercased, deduplicated
#    - Empty string → empty list
#    - All short words ("a to in") → empty list (all filtered)
#    - Duplicates removed ("the cat and the dog" → ["the", "cat", "and", "dog"] but "the" appears only once)
#    Wait: "the" is 3 chars so it IS included. Check: r"\b\w{3,}\b"

# 2. recall_by_document(driver, query, top_k):
#    - Mock the Neo4j driver at the SESSION level, not at some deep method
#    - Use a FakeSession that captures the Cypher and returns controlled rows
#    - Assert: empty query → [] (no DB call)
#    - Assert: top_k clamped to [1, 20]
#    - Assert: RecallHit objects have correct fields mapped from DB row

# 3. recall_by_dialog(driver, query, top_k):
#    - Same pattern: FakeSession, assert RecallHit fields

# 4. recall_by_topic(driver, query, top_k):
#    - Same pattern

# FakeSession pattern — mock at the right level:
class FakeRecord:
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key):
        return self._data[key]

class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
    def __iter__(self):
        return iter(self._rows)

class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.queries = []
        self.params = []
    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        self.params.append(kwargs)
        return FakeCursor(self._rows)
    def __enter__(self): return self
    def __exit__(self, *a): pass

class FakeDriver:
    def __init__(self, rows):
        self._rows = rows
        self.session_obj = FakeSession(rows)
    def session(self):
        return self.session_obj
```

**What to assert for recall functions:**
- Empty query returns `[]` without calling driver at all
- Non-empty query calls driver.session() and returns RecallHit objects
- top_k=0 is clamped to 1 (no crash)
- top_k=100 is clamped to 20
- RecallHit.text == row["text"], RecallHit.score == row["score"], etc.
- FakeSession rows with NO matching entity names → returns empty list (don't crash)

### File: `tests/test_context.py` (NEW)
Test `graphknows/retrieval/context.py`:

All these are pure Python dataclass + string generation — NO DB needed.

Key behaviors:
1. `RetrievalContext.to_dict()`:
   - Check structure: has keys query, session_id, search_type, graph_matches, semantic_matches, fused_results
   - graph_matches → list of dicts with name, label, similarity, chunks
   - Empty context → empty lists, not None

2. `RetrievalContext.to_xml()`:
   - Valid XML output (use `xml.etree.ElementTree.fromstring()` to parse it — if it throws, the XML is broken)
   - XSS injection in query/text: `query="<script>alert(1)</script>"` → escaped in output
   - XSS in chunk text: `text='<b onclick="evil()">'` → escaped
   - `total_chunks` attribute == len(fused_results)
   - `count` attribute in `<graph_matches>` == len(graph_matches)
   - `count` in `<semantic_matches>` == len(semantic_matches)
   - `count` in `<fused_results>` == len(fused_results)

3. `_esc()` helper:
   - `<`, `>`, `&`, `"` all properly escaped

4. `_metadata_tag()` helper:
   - Missing keys → empty string attributes (not crash)

5. `_mention_weight()`:
   - Falls back to entity_similarity when metadata missing

### File: `tests/test_graphgen_utils.py` (NEW)
Test pure utility functions in `graphknows/graphgen/utils/`:

#### From `utils.py`:
```python
# standardize_label:
# "Political Party" → "POLITICAL_PARTY"  
# "EU Commission" → "EU_COMMISSION"
# "" → ""  (empty string returns empty string, no crash)
# "Hello-World" → "HELLO_WORLD"
# "  spaces  " → ... what? strip + upper
# "abc" → "ABC"

# merge_node_into(graph, source, target):
# - Creates DiGraph with 3 nodes, source→target edge
# - After merge: source_node removed from graph
# - target_node still exists
# - Edges from source transferred to target
# - Self-loops handled (source→source transferred to target→target)
# - If source not in graph → no crash, no change

# create_output_directory: 
# - creates a dir in tmp, verify it exists
```

#### From `provenance.py`:
```python
# redact_secrets(data):
# - {"api_key": "SECRET123", "name": "Alice"} → {"api_key": "REDACTED", "name": "Alice"}
# - Nested: {"config": {"token": "abc"}} → {"config": {"token": "REDACTED"}}
# - List values: [{"api_key": "x"}] → [{"api_key": "REDACTED"}]
# - Non-sensitive dict unchanged
# - Key check is case-insensitive: "API_KEY", "Token", "SECRET_KEY" all redacted

# get_git_revision(repo_root):
# - Pass actual /root/graphknows → returns a string (sha or ref)
# - Pass non-existent dir → returns None, no crash
```

#### From `diagnostics.py`:
```python
# diagnostics_enabled(config):
# - None → False
# - {} → False
# - {"extraction": {"diagnostic_mode": True}} → True
# - {"extraction": {"diagnostic_mode": False}} → False
# - Works with Pydantic model objects too (via model_dump)

# diagnostic_dir(config):
# - {"infra": {"output_dir": "/tmp/out"}, "extraction": {"diagnostic_output_subdir": "diag"}}
#   → Path("/tmp/out/diag")
# - Missing keys → falls back to defaults ("output" / "diagnostics")

# _sanitize_name(name):
# - "Hello World!" → "Hello_World"
# - "a/b/c" → "a_b_c"
# - Empty → "diagnostic"
```

#### From `labels.py`:
```python
# resolve_entity_labels(extraction_config):
# - {} → returns DEFAULT_LABELS list (non-empty)
# - {"gliner_labels": ["Person", "Entity"]} → returns those labels (standardized)
# - No crash on missing keys
```

### File: `tests/test_graph_cleaning.py` (NEW)
Test `graphknows/graphgen/pipeline/graph_cleaning/canonicalization.py` and `pruning.py`.

Read these files first to understand their public API and write tests for the pure functions.
Focus on:
- Functions that take a networkx graph and transform it
- Functions that take strings and normalize/clean them
- Edge cases: empty graph, single node, disconnected graph

## Acceptance criteria

1. All new test files pass: `uv run pytest tests/test_recall_module.py tests/test_context.py tests/test_graphgen_utils.py tests/test_graph_cleaning.py -v --tb=short -m 'not integration'`
2. Coverage increases significantly (measured after: `uv run python -m pytest tests/ -q -m 'not integration' -p pytest_cov --cov=graphknows --cov-report=term-missing --tb=no 2>&1 | tail -20`)
3. Ruff passes: `uv run ruff check --config ruff.toml graphknows/ tests/` AND `uv run ruff format --check --config ruff.toml graphknows/ tests/`
4. `recall.py` coverage goes from 0% → ~90%+
5. `context.py` coverage goes from 40% → ~90%+
6. `graphgen/utils/*.py` goes from 0% → ~70%+

## Files to create
- `tests/test_recall_module.py` (NEW)
- `tests/test_context.py` (NEW)
- `tests/test_graphgen_utils.py` (NEW)
- `tests/test_graph_cleaning.py` (NEW)

## What NOT to do
- Do NOT use `@patch` to mock internal implementation details
- Do NOT write tests that only assert "no exception raised"
- Do NOT modify any production code
- Do NOT mock at a deeper level than the DB driver boundary
- Do NOT add `noqa` suppressions or `# type: ignore` without reason
- Do NOT modify existing test files
- Do NOT touch graphknows/ production code
- Do NOT assert on internal method call counts unless testing retry/error logic
