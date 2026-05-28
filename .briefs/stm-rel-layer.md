# Brief: STM Entity-Relation Layer

## Working directory
`/root/graphknows`

## Context
The graphknows memory system ingests text into Memgraph (STM) + pgvector.
GLiNER2 entity extraction is already working. GLiNER2 ALSO performs relation extraction
(the Schema API configures it with relation types), but the output is silently discarded.

### Files to touch (ONLY these):
- `graphknows/memory/extractor.py` — fix to return relation output alongside entities
- `graphknows/retrieval/stores/memgraph.py` — add a method to write REL edges
- `graphknows/memory/ingestor.py` — wire: after entity extraction, write REL edges to Memgraph

### Files NOT to touch:
Everything else. No refactors, no style fixes elsewhere.

## The fix

### Step 1: `graphknows/memory/extractor.py`

In `GLiNER2EntityExtractor.extract()`, the current code does:
```python
raw = self._model.predict_relations(...)
entities = raw.get("entities", [])
# relations are in raw.get("relation_extraction", []) but are never read
return [{"name": e["text"], "type": e["label"], "score": e["score"]} for e in entities]
```

Change the return type to a named tuple or dataclass with two fields:
```python
@dataclasses.dataclass
class ExtractionResult:
    entities: list[dict]   # [{name, type, score}]
    relations: list[dict]  # [{head, head_type, relation, tail, tail_type, score}]
```

Read `raw.get("relation_extraction", [])` and return it.

The relation output from GLiNER2 Schema API looks like:
```json
[{"head": "Viktor Orbán", "head_type": "PERSON",
  "relation": "OPPOSES", "tail": "Ukraine EU membership",
  "tail_type": "CONCEPT", "score": 1.0}]
```

**Important**: The extractor is called from `MemoryIngestor._ingest_parsed()` and from tests.
Check all call sites and update them.

### Step 2: `graphknows/retrieval/stores/memgraph.py`

Add a new method `write_relation_edge()`:

```python
def write_relation_edge(
    self,
    session_id: str,
    subject_name: str,
    relation_name: str,
    object_name: str,
    fact: str,
    confidence: float,
    valid_at: datetime,
    created_at: datetime,
) -> None:
    """Write a semantic relation edge between two entities in STM.

    Uses generic :REL label with a 'name' property to avoid Memgraph
    dynamic edge label constraints. Query by property: MATCH ()-[r:REL {name: 'OPPOSES'}]->()
    """
```

The Cypher should:
1. MATCH the two Entity nodes by name AND session_id
2. MERGE a `:REL` edge between them
3. SET all properties: name, fact, confidence, valid_at, created_at

**Memgraph pitfalls (mandatory reading)**:
- Do NOT use dict properties — flatten to individual SET clauses
- Do NOT use `CALL db.labels()` — it doesn't exist
- Match entities by `e.name = $name AND e.session_id = $session_id`
- Use `id(n)` not `elementId(n)`

```cypher
MATCH (s:Entity {name: $subject, session_id: $session_id})
MATCH (o:Entity {name: $object, session_id: $session_id})
MERGE (s)-[r:REL {name: $name, session_id: $session_id}]->(o)
SET r.fact = $fact,
    r.confidence = $confidence,
    r.valid_at = $valid_at,
    r.created_at = $created_at
```

### Step 3: `graphknows/memory/ingestor.py`

In `MemoryIngestor._ingest_parsed()`, after the entity extraction + write loop, add:

```python
# Write relation edges to STM
for rel in extraction_result.relations:
    subject = rel.get("head", "")
    object_ = rel.get("tail", "")
    relation_name = rel.get("relation", "")
    score = rel.get("score", 0.0)
    if subject and object_ and relation_name:
        fact = f"{subject} {relation_name.lower().replace('_', ' ')} {object_}"
        self._mg_store.write_relation_edge(
            session_id=session_id,
            subject_name=subject,
            relation_name=relation_name,
            object_name=object_,
            fact=fact,
            confidence=score,
            valid_at=now,
            created_at=now,
        )
```

## Tests

Add/update tests in `tests/memory/test_extractor.py`:
- Test that `ExtractionResult` has both `entities` and `relations` fields
- Mock GLiNER2 model to return both entity and relation output
- Verify relations are returned

Add/update tests in `tests/retrieval/test_memgraph_store.py`:
- Test `write_relation_edge()` generates correct Cypher
- Mock at `session.run()` level, assert Cypher string contains `MERGE`, `:REL`, `name`, `fact`

Add/update integration test marker in `tests/memory/test_ingestor.py` if it exists:
- After ingest, assert that relations are attempted to be written

## Acceptance criteria
1. `ExtractionResult` dataclass exported from `graphknows.memory.extractor`
2. `GLiNER2EntityExtractor.extract()` returns `ExtractionResult` with both fields populated
3. `MemgraphStore.write_relation_edge()` method exists and generates correct Cypher
4. `MemoryIngestor._ingest_parsed()` calls `write_relation_edge()` for each extracted relation
5. All existing tests still pass: `uv run pytest tests/ -x -q -m "not integration"`
6. New tests for extractor and memgraph store pass

## End-of-task summary format
```
Files changed: <list>
What was modified: <one line each>
Files not touched: confirmed
Tests: uv run pytest tests/ -x -q -m "not integration" → EXIT CODE + summary line
Follow-up: <list or none>
```

## Coding standards
Read before writing any code:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

Key rules:
- Type annotations on all public APIs
- Google docstrings on all public modules, classes, functions  
- TDD: write failing test first
- Only touch the three files listed above
- Do NOT commit
