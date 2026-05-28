# Memory Layer — Full Bug Discovery Audit

## Your job
Do a **read-only audit** of the GraphKnows memory layer code and tests.
Do NOT modify any code. Do NOT commit anything.
Write all findings to `/root/graphknows/research/mem_audit_report.md`.

## Confirmed working directory
`/root/graphknows/services/graphknows`

## Context — what the memory layer does
The memory layer (implemented during the last sprint) provides:
- **STM** (short-term, session-scoped): Memgraph graph + pgvector stm_* tables
- **LTM** (long-term, cross-session): Neo4j graph + pgvector ltm_* tables
- **CLI**: `graphknows-memory` commands (ingest, flush, recall-short, recall-long, stats, forget, doctor)

Key files:
- `graphknows/memory/__init__.py`, `ingestor.py`, `dialog.py`, `flush.py`
- `graphknows/retrieval/recall_stm.py`, `recall_ltm.py`
- `graphknows/retrieval/stores/memgraph.py`, `pgvector.py`
- `graphknows/ingestion/parsers/base.py`, `text.py`, `code.py`, `docx.py`, `registry.py`
- `graphknows/cli/memory.py`
- `graphknows/schema.py`
- `tests/test_schema.py`, `test_parsers.py`, `test_memory_ingestor.py`,
  `test_dialog_writer.py`, `test_flush.py`, `test_recall.py`, `test_cli_memory.py`

## Known bugs (already ticketed — still investigate root cause and exact line)
1. **ENTITY.type = None** — every entity written to Memgraph has `type=None`. The `type` property is not being set. Find the exact write site and the field name mismatch.
2. **IN_SESSION edges missing** — FILE and DIALOG nodes have `session_id` as a property but are NOT connected to their SESSION node via `IN_SESSION`. Find the Cypher MERGE that should create `(SESSION)-[:IN_SESSION]->(FILE)` and `(SESSION)-[:IN_SESSION]->(DIALOG)` and identify why it is not running.
3. **TURN -[:NEXT]-> TURN chain not written for new sessions** — 5 NEXT edges exist in Memgraph but none belong to the most recent demo session. Find the write site in `dialog.py` or `memgraph.py`.

## What to audit — be exhaustive

### A. Schema correctness
- Does `schema.py` define all required node labels and relationship types?
- Are Cypher constraints and indexes defined for SESSION, FILE, CHUNK, ENTITY, DIALOG, TURN?
- Does the pgvector DDL create all 5 tables: stm_chunks, stm_turns, ltm_chunks, ltm_turns, ltm_topics?
- Is there any mismatch between the schema definition and what the stores actually write?

### B. Memgraph store (`retrieval/stores/memgraph.py`)
- Trace every `write_*` method: does it set all properties it claims to? Check field names carefully.
- Does `write_entity_node` set `type`? Where does it get the value from?
- Does `write_file_node` create `(SESSION)-[:IN_SESSION]->(FILE)`?
- Does `write_dialog_node` create `(SESSION)-[:IN_SESSION]->(DIALOG)`?
- Does `write_turn_sequence` actually get called? From where? Does it run per-session?
- Are all async methods properly awaited upstream?
- Is there any exception swallowing (`try/except pass` or `except Exception: logger.warning`) that silently drops writes?

### C. Ingestor (`memory/ingestor.py`)
- Trace the full ingest path: text → chunks → entities → writes.
- Does it pass `entity.type` (or equivalent field) to the store's `write_entity_node`?
- Does it call `write_file_node` with the session_id? Does that call create the IN_SESSION edge?
- Are chunk embeddings always computed? What happens on embedding failure?
- Is the `_BG_LOOP` pattern correct? Any race conditions or loop attachment issues?
- Does every store call check for errors or swallow them?

### D. Dialog writer (`memory/dialog.py`)
- Does it call `write_dialog_node` with session_id? Does that create IN_SESSION edge?
- Does it call `write_turn_sequence` for each consecutive turn pair?
- Are TURN→ENTITY MENTIONS edges written?
- What happens when entity extraction fails on a turn?

### E. Flush (`memory/flush.py`)
- Is the flush idempotent (MERGE semantics in Neo4j)?
- Does it correctly read from stm_chunks/stm_turns and write to ltm_chunks/ltm_turns?
- Does it create TOPIC nodes and run community detection?
- Does it clear/expire STM after flush, or is that left to the caller?
- Any missing error handling?

### F. Recall (`retrieval/recall_stm.py`, `recall_ltm.py`)
- Does RRF correctly combine chunk + turn scores?
- Is the entity-boost path correctly reading entities from Memgraph?
- Does `entity_names` always propagate into `RecallResult`?
- Any edge cases: empty results, no embedding, missing session?
- LTM recall: does it gracefully handle empty Neo4j (no ENTITY label yet)?

### G. Parsers (`ingestion/parsers/`)
- Does `PlainTextParser` preserve heading hierarchy correctly?
- Does `CodeParser` handle all registered extensions?
- Does `DocxParser` handle missing python-docx gracefully?
- Is the registry dispatch correct for all file extensions?
- Any missing edge cases: empty input, binary content, very long text?

### H. CLI (`cli/memory.py`)
- Are all commands wired up: ingest, flush, recall-short, recall-long, stats, forget, doctor?
- Does `doctor` check all three stores (Memgraph, pgvector, Neo4j)?
- Does `forget` clean up all three stores atomically, or partially?
- Any missing error messages or unhelpful exceptions for bad inputs?

### I. Tests
- Do the tests actually cover the known bugs? (entity type, IN_SESSION edges, NEXT chain)
- Are there any tests using `MagicMock` where `AsyncMock` is needed?
- Are there missing test cases for: flush idempotency, entity merge, topic creation, recall with no results?
- Is the CI gate complete? What tests are currently ignored and why?

### J. Cross-cutting concerns
- Is there any code that catches broad `Exception` and logs/ignores it, hiding real failures?
- Are there any hardcoded session IDs, test fixture leaks, or shared mutable state between tests?
- Is the async/sync boundary (the `_BG_LOOP`/`_run` pattern) consistent across all callers?
- Any import cycles or circular dependencies?

## Output format
Write `/root/graphknows/research/mem_audit_report.md` with:

```markdown
# Memory Layer Audit Report

## Summary
<short paragraph: overall health, critical vs minor issues>

## Critical bugs (data loss / incorrect data)
For each: file + line number, description, impact, suggested fix

## Minor bugs (incorrect behaviour, edge cases)
For each: file + line number, description, impact, suggested fix

## Test gaps
For each: what is untested, why it matters, suggested test

## Code quality issues
For each: file + location, description (but only if it materially affects correctness or maintainability)

## Clean / working correctly
List what you verified is actually correct — do not only list problems.
```

Be precise. Include file paths and line numbers where possible. Do not pad the report with generic observations.
