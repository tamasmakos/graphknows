# Memory Layer — 3 Critical Bug Fixes

## Working directory
`/root/graphknows` (run all commands from here)

## PYTHONPATH for tests
```bash
cd /root/graphknows && set -a && source .env && set +a
export PYTHONPATH=/root/graphknows/services/graphknows
```

## The 3 bugs to fix (all in the same files — do them in one pass)

---

### FIX 1 — `ENTITY.type` property name mismatch
**File:** `services/graphknows/graphknows/retrieval/stores/memgraph.py`

**Line 190:** The Cypher writes `e.entity_type` but all queries read `e.type`.

Change:
```cypher
SET e.entity_type = $etype,
```
To:
```cypher
SET e.type = $etype,
```

Also check `services/graphknows/graphknows/retrieval/stores/neo4j.py` for the LTM equivalent — if it also writes `entity_type` instead of `type`, fix it there too.

---

### FIX 2 — Wrong relationship types: `HAS_FILE` / `HAS_DIALOG` → `IN_SESSION`

**File:** `services/graphknows/graphknows/retrieval/stores/memgraph.py`

**Line ~106 in `write_file_node`:**
```cypher
MERGE (sess)-[:HAS_FILE]->(f)
```
Change to:
```cypher
MERGE (sess)-[:IN_SESSION]->(f)
```

**Line ~259 in `write_dialog_node`:**
```cypher
MERGE (sess)-[:HAS_DIALOG]->(d)
```
Change to:
```cypher
MERGE (sess)-[:IN_SESSION]->(d)
```

---

### FIX 3 — `TURN.session_id` not stored

**File:** `services/graphknows/graphknows/retrieval/stores/memgraph.py`

The `write_turn_node` method (around line 268) does NOT set `session_id` on the TURN node.
The current Cypher:
```cypher
MERGE (t:TURN {id: $tid})
SET t.dialog_id = $did,
    t.turn_index = $idx,
    t.role = $role,
    t.text = $text,
    t.timestamp = $ts
```

Fix:
1. Add `session_id: str` parameter to `write_turn_node` signature (after `dialog_id`)
2. Add `t.session_id = $sid` to the SET clause
3. Pass `sid=session_id` to the `s.run()` call

**File:** `services/graphknows/graphknows/memory/dialog.py`

Find the call to `self._mg.write_turn_node(...)` and add `session_id=self._session_id` to the call. The `DialogWriter` class already has `self._session_id` available (it's set in `__init__` or `start_dialog`).

---

## Tests to update

After the code fixes, update the existing tests so they cover the new correct behaviour:

**`tests/test_dialog_writer.py`:**
- `test_write_turn_calls_write_turn_node`: assert that `write_turn_node` is called with `session_id` parameter

**`tests/test_memory_ingestor.py`:**
- `test_ingest_parsed_calls_write_entity_node_per_entity`: assert that entity type is passed (the call already passes it — just verify the mock captures it)

Add 3 new tests (one per bug) that assert the Cypher strings or method call args contain the corrected values:

1. **`test_write_entity_node_sets_type_not_entity_type`** — inspect that the Cypher string in `write_entity_node` contains `e.type` not `e.entity_type`. You can do this by reading the source file and asserting the string is present, or by using a fake session that captures the query.

2. **`test_write_file_node_uses_in_session_rel`** — same pattern, assert `IN_SESSION` appears in the Cypher string in `write_file_node`.

3. **`test_write_turn_node_stores_session_id`** — assert `session_id` is a parameter of `write_turn_node` and appears in the Cypher SET clause.

The simplest approach for these 3 tests is to read the source and do a string assertion:
```python
import inspect
from graphknows.retrieval.stores.memgraph import MemgraphStore
src = inspect.getsource(MemgraphStore.write_entity_node)
assert "e.type" in src and "e.entity_type" not in src
```

---

## CI gate (must pass before finishing)

```bash
cd /root/graphknows && set -a && source .env && set +a
export PYTHONPATH=/root/graphknows/services/graphknows

python3 /root/graphknows/scripts/check_imports.py
ruff format --check services/graphknows/graphknows/
ruff check --no-fix services/graphknows/graphknows/
python3 -B -m pytest tests/ -x -q -m "not integration" \
  --ignore=tests/test_ontology.py \
  --ignore=tests/test_shared_config.py \
  --ignore=tests/test_pgvector_retrieval.py \
  --ignore=tests/test_langgraph_agent_search_eu.py \
  --ignore=tests/test_ontology_classifier.py \
  --ignore=tests/test_ontology_extractor.py \
  --ignore=tests/test_runner_dry_run.py
```

All checks must be green. Do not commit.

## What NOT to touch
- Do not modify parsers, flush, recall, CLI, or any file not listed above.
- Do not refactor anything outside the 3 bug fixes.
- Do not commit or push.
