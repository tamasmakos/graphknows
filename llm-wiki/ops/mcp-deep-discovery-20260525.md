---
title: MCP Deep Discovery — 2026-05-25
created: 2026-05-25
updated: 2026-05-25
type: ops
tags: [mcp, bugs, schema, discovery, memory]
confidence: high
---

# MCP Deep Discovery — 2026-05-25

Systematic end-to-end probe of all 12 MCP tools, schema, and OpenCode integration.

## System State at Discovery Start

- Backends: Memgraph ✓, Neo4j ✓, pgvector ✓
- MCP version: `1a5dc3a` (matches HEAD)
- STM: 1 chunk, 0 turns
- LTM: 6 sessions, 20 entities, 0 topics, 8 chunks, 0 turns
- Stale STM Memgraph: 9 SESSION nodes only (all previously flushed or stale)
- LTM Neo4j: 20 ENTITY, 12 CHUNK, 6 SESSION, 6 FILE, 4 PASSAGE, 2 DOC, 1 SEGMENT, 1 REL edge

## Bugs Found (Priority Order)

### BUG-001 — CRITICAL: `memory_ingest` crashes immediately (FIXED)

**Error:** `MemoryIngestor.__init__() got an unexpected keyword argument 'rel_extractor'`

**Root cause:** `stdio_server.py` passes `rel_extractor=rel_extractor` to `MemoryIngestor.__init__()` (line 297),
but `ingestor.py` `__init__` signature only accepts `extractor`, `embed_fn`. The kwarg was added to the server
side for REL-003 but never added to the ingestor constructor.

**Fix applied:** Added `rel_extractor: Any | None = None` to `MemoryIngestor.__init__()` and stored as `self._rel_extractor`.

**Note:** `self._rel_extractor` is now stored but still not used in `_ingest_parsed`. REL-003 is incomplete —
DSPy relation extraction is wired in the server but never called from ingest pipeline. Tracked as BUG-006.

---

### BUG-002 — HIGH: `## Content\n\n` prefix pollutes all stored text (FIXED)

**Symptom:** `PlainTextParser.parse()` produces `seg.text = '## Content\n\nActual text...'` for every segment.
This prefix was being stored in Memgraph, pgvector, and used for embedding — diluting embedding quality.

**Root cause:** The `_clean()` fix was only applied to the MCP recall formatter output, not at ingest time.
The ingestor wrote the raw `seg.text` (with prefix) to both Memgraph and pgvector.

**Fix applied:** In `ingestor._ingest_parsed()`: strip `## Content\n\n` prefix into `clean_text` before
extracting entities, computing embedding, and writing to Memgraph/pgvector. Both stores now receive `clean_text`.

**Effect:** All existing stored chunks contain the polluted text. Fresh ingests will be clean.
**Follow-up needed:** Consider a one-time migration to strip existing chunks.

---

### BUG-003 — HIGH: `ltm_query` queries non-existent DB columns (FIXED)

**Error:** `column "community_id" does not exist` (PostgreSQL)

**Root cause:** `stdio_server.ltm_query()` queried `community_id, entity_count, hub_entities` from `ltm_topics`,
but the actual schema has `topic_id, summary, entity_names, session_ids, created_at, embedding`.
The `community_id`-based schema was from an earlier iteration that was never reconciled.

**Fix applied:** Updated the SELECT to `topic_id, summary, entity_names, session_ids` with JSON deserialization
for array columns. Return shape updated to match: `{topic_id, summary, entity_names, session_ids}`.

---

### BUG-004 — HIGH: `recall_by_document`, `recall_by_dialog`, `recall_by_topic` crash on import (FIXED)

**Error:** `ModuleNotFoundError: No module named 'graphknows.recall'`

**Root cause:** All three MCP tools import from `graphknows.recall` which was deleted from the repo
(only a stale `.pyc` cache remained, which Python cannot use as a live module).

**Fix applied:** Created `/root/graphknows/graphknows/recall.py` with `RecallHit` dataclass and three
entity-graph-based recall functions using Neo4j Cypher traversal:
- `recall_by_document(driver, query, top_k)` — chunk nodes ranked by entity overlap
- `recall_by_dialog(driver, query, top_k, session_filter)` — TURN nodes ranked by entity match
- `recall_by_topic(driver, query, top_k)` — Topic nodes ranked by entity membership

---

### BUG-005 — MEDIUM: Segment metadata is always `{}` (no fix applied yet)

**Symptom:** All stored chunks have `metadata = {}` in pgvector.

**Root cause:** `PlainTextParser` and `CodeParser` produce `seg.metadata = {}` — no metadata is generated
by any parser. The `{**seg.metadata, **metadata}` in the ingestor correctly passes call-site metadata,
but call sites (especially the MCP `memory_ingest` tool) don't pass useful metadata either.

**Missing metadata fields that would be useful:**
- `atom_type` (text/code/terminal)
- `source_url` / `origin`
- `language` (for code segments)
- `memory_class` (raw_evidence / state / fact / episode)
- `created_at` (ISO timestamp)
- `word_count` / `char_count`
- `parser_name` (already in `ParsedFile.parser_name` but not propagated to segments)

**Impact:** Cannot filter/boost by type during recall. Provenance is opaque.
**Action needed:** Parsers should populate `seg.metadata` with at minimum `atom_type` and `parser_name`.
MCP `memory_ingest` should accept and forward a `metadata` dict param.

---

### BUG-006 — MEDIUM: REL-003 incomplete — `_rel_extractor` stored but never called (not fixed)

**Symptom:** `OntologyRelExtractor` is wired in `stdio_server.py` when `OPENROUTER_API_KEY` is set,
but even after BUG-001 fix, the ingestor never calls it. `_rel_extractor` is stored as `self._rel_extractor`
but `_ingest_parsed()` has no code to invoke it.

**What's missing:** After entity extraction, call `self._rel_extractor.extract_relations(entities, text)`
and write resulting relations as `:REL` edges in Memgraph via `write_relation_edge()`.

**The extractor also drops its own relation output:** `GLiNER2EntityExtractor.extract()` calls `self._gliner.extract()`
which returns a dict with BOTH `entities` AND `relation_extraction` keys. Only `entities` is parsed and returned.
The `relation_extraction` output (GLiNER2's built-in relation model) is silently discarded.

**Action needed:**
1. Add `extract_with_relations(text)` method to `GLiNER2EntityExtractor` that returns both entities and GLiNER2 relations
2. In `_ingest_parsed`, after entity extraction, call GLiNER2 relation extraction and write `:REL` edges
3. If `_rel_extractor` is set, also run DSPy pass and merge results

---

### BUG-007 — LOW: `ltm_topics` MCP tool queries `t.community_id` in Neo4j (may fail when topics exist)

**Symptom:** `ltm_topics` tool returns `{}` (no topics) now because DB is empty. When topics ARE populated
via the batch graphgen pipeline (which uses `community_id`), this would work. But if topics come from the
memory flush pipeline (`upsert_topics` in pgvector using `topic_id`), Neo4j and pgvector schemas diverge.

**Note:** The Neo4j `Topic` nodes written by `neo4j.py::upsert_topics` use `community_id` as their property.
The pgvector `ltm_topics` uses `topic_id`. These are different data stores — the MCP `ltm_topics` tool queries
Neo4j correctly for Neo4j topics. `ltm_query` queries pgvector — that was the broken one (now fixed).

---

## Schema Analysis

### STM (Memgraph + pgvector `stm_*`)

**Memgraph nodes:** SESSION, FILE, CHUNK, ENTITY, DIALOG, TURN
**Memgraph edges:** IN_SESSION (FILE→SESSION), HAS_CHUNK (FILE→CHUNK), HAS_ENTITY (CHUNK→ENTITY), NEXT (TURN→TURN)
**pgvector stm_chunks:** id, session_id, file_id, text, heading_path, embedding, metadata, created_at
**pgvector stm_turns:** id, session_id, dialog_id, turn_index, role, text, embedding, created_at

**Missing from STM schema:**
- `metadata.atom_type` — what kind of content (document/code/terminal/dialog)
- `metadata.memory_class` — (raw_evidence/state/fact/episode) for recall boosting
- `metadata.source_url` — provenance URL
- `metadata.word_count` — for chunk sizing context
- No `provenance.memory_class` column on `stm_chunks` (mentioned in skill as gap)
- ENTITY nodes: `session_id` is on entities in Memgraph but entities are merged by name in Neo4j LTM — 
  cross-session entity identity is correct in LTM but STM entities can't be queried cross-session easily

### LTM (Neo4j + pgvector `ltm_*`)

**Neo4j nodes:** SESSION, FILE, CHUNK, ENTITY, PASSAGE, DOC, SEGMENT, TOPIC
**Neo4j edges:** IN_SESSION, HAS_CHUNK, HAS_ENTITY, REL (entity-entity), COMPONENT_OF (ENTITY→TOPIC)
**pgvector ltm_chunks:** id, session_id, file_id, text, heading_path, embedding, metadata, flushed_at
**pgvector ltm_turns:** id, session_id, dialog_id, turn_index, role, text, embedding, flushed_at
**pgvector ltm_topics:** topic_id, summary, embedding, entity_names, session_ids, created_at

**Missing from LTM schema:**
- `ltm_topics` has no `entity_count` column despite `memory_stats` tracking it via `count(*)` approximation
- `ltm_turns` has no `metadata` column (unlike `ltm_chunks` which has it)
- ENTITY nodes in Neo4j have no `first_seen_at` / `last_seen_at` (mentioned in `ltm_entity` docstring but not written)
- Bi-temporal edge support (`valid_from`, `valid_to`, `created_at`, `expired_at`) not implemented
- `provenance.memory_class` not stored anywhere in LTM

### OpenCode Integration

**Tools available:** graphknows-ingest, graphknows-recall, graphknows-flush, graphknows-entity
**Pattern:** Each tool spawns `graphknows-mcp` subprocess, sends MCP JSON-RPC via stdin, reads last line
**Assessment:** Pattern is correct but fragile — spawns a new process per call (15s GLiNER2 cold start hit every time)
**Improvement needed:** Consider a persistent MCP daemon or OpenCode session-level tool initialization

## Recommendations (Priority Order)

1. **URGENT:** Verify all 4 bug fixes work end-to-end after MCP reload
2. **HIGH:** Wire GLiNER2 relation extraction in `GLiNER2EntityExtractor.extract()` (currently discarded)
3. **HIGH:** Add `atom_type`, `memory_class`, `parser_name` to segment metadata in parsers
4. **HIGH:** Add `metadata` dict param to MCP `memory_ingest` tool for caller-provided provenance
5. **MEDIUM:** Add `memory_class` column to `stm_chunks` and `ltm_chunks` pgvector tables
6. **MEDIUM:** Wire `_rel_extractor` (DSPy) in `_ingest_parsed` after entity extraction
7. **MEDIUM:** OpenCode warm-start: document that each tool call incurs 15s cold start if process not cached
8. **LOW:** Migration script to strip `## Content\n\n` from existing stored chunks
9. **LOW:** Add `first_seen_at`/`last_seen_at` to ENTITY nodes in Neo4j
10. **LOW:** Consolidate `ltm_topics` Neo4j vs pgvector schemas into a unified design

## Related

- [[architecture/memory-layer-v2]]
- [[ops/technical-debt]]
- [[architecture/mcp-server-design]]
