# LTM-8: Schema Migration — DOCUMENT, CHUNK, DIALOG, UTTERANCE

## Context

GraphKnows uses Neo4j as its long-term memory (LTM) store. The current schema has:
- `Entity` nodes with PageRank, community_id, node2vec embeddings
- `Segment` nodes (one per session writeback) with `APPEARED_IN` edges to Entities
- `Topic` / `SubTopic` nodes from Louvain community detection
- `GDSRun` nodes with quality metrics per session

The user wants to add two new memory layers:
1. **Document memory** — structured ingestion of PDF/URL sources with chunk-level provenance
2. **Dialog memory** — conversation turns (utterances) linked to entities mentioned

This task is **schema + plumbing only**. No data ingest pipelines yet.

## Target Schema

```
(Document {id, title, url, source_type, ingested_at})
    -[HAS_CHUNK]->
(Chunk {id, text, char_offset, token_count, embedding: float[]})
    -[MENTIONS]->
(Entity {name, type, ...})  ← already exists

(Session {id, started_at, ended_at})          ← currently as Segment label, keep both
    -[HAS_DIALOG]->
(Dialog {id, session_id, started_at})
    -[PARTICIPANT]->
(Participant {id, role: "agent"|"user", name})
    -[SAID]->
(Utterance {id, text, turn_index, timestamp, embedding: float[]})
    -[MENTIONS]->
(Entity)

(Utterance) -[REFERENCES]-> (Chunk)           ← optional, for grounded citations
```

All embedding arrays are `float[]` (same as existing `node2vec` property on Entity).

## Files to Create/Modify

### 1. NEW: `services/graphknows/graphknows/schema.py`

A module that owns all schema management — constraints, indexes, and migration.

```python
"""Neo4j schema: constraints, indexes, and migration for LTM nodes."""

CONSTRAINTS = [
    # Existing (idempotent — skip if already exist)
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id_unique IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.community_id IS UNIQUE",
    # New
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:Dialog) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
]

INDEXES = [
    # Existing (idempotent)
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    # New — full-text for dialog/chunk search
    "CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX utterance_text_idx IF NOT EXISTS FOR (u:Utterance) ON EACH [u.text]",
    # Range indexes for temporal queries
    "CREATE INDEX document_ingested_idx IF NOT EXISTS FOR (d:Document) ON (d.ingested_at)",
    "CREATE INDEX utterance_timestamp_idx IF NOT EXISTS FOR (u:Utterance) ON (u.timestamp)",
    "CREATE INDEX dialog_session_idx IF NOT EXISTS FOR (d:Dialog) ON (d.session_id)",
]

def apply_schema(session) -> dict:
    """Apply all constraints and indexes. Idempotent. Returns counts."""
    applied = {"constraints": 0, "indexes": 0, "errors": []}
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
            applied["constraints"] += 1
        except Exception as e:
            applied["errors"].append(f"constraint: {e}")
    for stmt in INDEXES:
        try:
            session.run(stmt)
            applied["indexes"] += 1
        except Exception as e:
            applied["errors"].append(f"index: {e}")
    return applied
```

### 2. MODIFY: `services/graphknows/graphknows/forget.py`

The `ForgetService.prune()` method currently only handles `Entity`, `Segment`, `Topic`, `SubTopic`, `GDSRun` nodes. Extend it to also prune:
- `Document` nodes with no connected `Chunk` nodes (orphaned docs)
- `Chunk` nodes with no `MENTIONS` edges (unlinked chunks)
- `Dialog` nodes with no `Utterance` children (empty dialogs)
- `Utterance` nodes with no `SAID` parent (orphaned utterances)

Add a TTL prune for `Utterance` nodes older than `utterance_ttl_days` (default 180 days).

Pattern for each — add after the existing `APPEARED_IN` prune block:

```python
# Prune orphaned Document nodes
s.run("""
    MATCH (d:Document)
    WHERE NOT (d)-[:HAS_CHUNK]->()
    DELETE d
""")

# Prune orphaned Chunk nodes
s.run("""
    MATCH (c:Chunk)
    WHERE NOT ()-[:HAS_CHUNK]->(c) OR NOT (c)-[:MENTIONS]->()
    DELETE c
""")

# Prune empty Dialog nodes
s.run("""
    MATCH (d:Dialog)
    WHERE NOT ()-[:SAID]->(:<any>)-[:*]->(d)
    AND NOT (d)-[:SAID]->()
    DELETE d
""")

# TTL prune for old Utterances
s.run("""
    MATCH (u:Utterance)
    WHERE u.timestamp < $cutoff
    DETACH DELETE u
""", cutoff=cutoff_iso)
```

### 3. MODIFY: `clean_ltm_graph.py`

Add cleanup for new node types to the "wipe Topics/SubTopics/GDSRun" section.
Keep Documents, Chunks, Dialogs, Utterances when doing a test reset — just wipe the computed labels.

Add a `--wipe-documents` flag that also wipes Document+Chunk+Dialog+Utterance (for full reset).

### 4. NEW: `services/graphknows/graphknows/migrations/001_ltm8_schema.py`

A standalone migration script:
```python
"""Migration 001 — LTM-8 schema: Document, Chunk, Dialog, Utterance."""
from graphknows.schema import apply_schema
from graphknows.config import get_neo4j_driver

def run():
    driver = get_neo4j_driver()
    with driver.session() as s:
        result = apply_schema(s)
        print(f"Applied {result['constraints']} constraints, {result['indexes']} indexes")
        if result['errors']:
            print("Errors:", result['errors'])

if __name__ == "__main__":
    run()
```

### 5. UPDATE: `services/graphknows/graphknows/writeback.py` (or wherever `merge_session_to_neo4j` lives)

After the Entity/Relation merge, call `apply_schema()` once at startup (not per-session). 
Add a guard at the top of the Neo4j driver init:

```python
_schema_applied = False

def ensure_schema(driver):
    global _schema_applied
    if not _schema_applied:
        with driver.session() as s:
            apply_schema(s)
        _schema_applied = True
```

## Tests to Add

In `test_ltm_pipeline.py`, add a new check section `── Schema Checks ──`:

```python
# Verify constraints exist
constraints = s.run("SHOW CONSTRAINTS").data()
constraint_names = {c.get('name','') for c in constraints}
schema_checks = [
    ('Document constraint', any('document' in n.lower() for n in constraint_names)),
    ('Chunk constraint', any('chunk' in n.lower() for n in constraint_names)),
    ('Dialog constraint', any('dialog' in n.lower() for n in constraint_names)),
    ('Utterance constraint', any('utterance' in n.lower() for n in constraint_names)),
]
```

And a smoke test for prune with a dangling Document:
```python
# Create orphan Document, run prune, verify it's gone
s.run("CREATE (:Document {id: 'orphan-doc-test', title: 'test', ingested_at: datetime()})")
forget_service.prune(session_id=None)
count = s.run("MATCH (d:Document {id: 'orphan-doc-test'}) RETURN count(d) as c").single()["c"]
assert count == 0, "orphan Document should have been pruned"
```

## Acceptance Criteria

1. `schema.py` exists with all constraints and indexes listed above
2. `migrations/001_ltm8_schema.py` runs successfully (idempotent — safe to run twice)
3. `forget.py` prunes orphan Document, Chunk, Dialog, Utterance nodes
4. `clean_ltm_graph.py` has `--wipe-documents` flag
5. `test_ltm_pipeline.py` schema checks: all 4 new constraints visible in `SHOW CONSTRAINTS`
6. Orphan Document prune smoke test passes
7. All existing 5/5 checks still pass
8. Commit with message: `LTM-8: schema migration — Document, Chunk, Dialog, Utterance + constraints + forget pruning`

## Constraints & Conventions

- Python 3.11, strict type hints, Google docstrings, 100-char lines
- No bare except — catch specific exceptions
- All Cypher: use `IF NOT EXISTS` on CREATE CONSTRAINT/INDEX (idempotent)
- All Neo4j driver calls: use `q=query` not `query=query` (driver reserves `query` as positional)
- f-strings in all Cypher segments that use `{{` (GDS pitfall — every continuation string must be f-string)
- Run with: `cd /root/graphknows && /root/graphknows/services/agent/.venv/bin/python -B migrations/001_ltm8_schema.py`
- Test with: `cd /root/graphknows && /root/graphknows/services/agent/.venv/bin/python -B test_ltm_pipeline.py`
- PYTHONPATH: `/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows`
- .env at `/root/graphknows/.env` — source it before running
- Do NOT commit. Leave for principal-engineer review.

## Known Facts (pre-investigated)

- Neo4j: bolt://172.18.0.1:7687, credentials in `/root/graphknows/.env`
- Current constraints: Entity name, Segment id, Topic community_id (verify with `SHOW CONSTRAINTS`)
- Current labels in use: Entity, Segment, Topic, SubTopic, GDSRun, Participant (maybe)
- `apply_schema()` pattern: neo4j driver session, `session.run(stmt)`, catch `ClientError` for "already exists" (idempotent with IF NOT EXISTS)
- `forget.py` prune currently handles: disconnected Entity components (pagerank < cutoff), old Segments (TTL), orphan Topic/SubTopic nodes
- `clean_ltm_graph.py` wipes: Topic, SubTopic, GDSRun, community_id property — preserves Entity + Segment
- Project root: `/root/graphknows`, branch: `main`, commit: `6c50bcf`
