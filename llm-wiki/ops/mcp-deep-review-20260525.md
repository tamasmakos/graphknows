# MCP Deep Review — 2026-05-25

Principal engineer audit of all 12 tools. Every finding backed by live DB evidence.

## Live DB State at Review Time

| Store | Count |
|---|---|
| Memgraph SESSION nodes | 11 |
| Memgraph TURN nodes | **0** |
| Memgraph ENTITY nodes | **0 (in STM)** |
| Memgraph CHUNK→ENTITY edges | **0** |
| pgvector stm_chunks | 1 |
| pgvector ltm_chunks | 11 |
| pgvector stm/ltm turns | 0 / 0 |
| pgvector ltm_topics | **0** |
| Neo4j SESSION nodes | 8 |
| Neo4j ENTITY nodes | 36 |
| Neo4j Topic nodes | **0** |
| Neo4j REL edges | 1 |

---

## Tool-by-Tool Findings

### 1. `memory_ingest`

**Status:** ⚠️ Degraded — works but with two silent data quality bugs

**Findings:**
- A. `PlainTextParser` wraps every chunk as `"## Content\n\ntext"` before calling parse. The `_re.sub` in `ingestor._ingest_parsed()` strips the prefix from `clean_text` for embedding/extraction, but `heading_path` on the pgvector chunk row still contains `"## Content"`. This is cosmetically ugly but not blocking.
- B. `seg.metadata = {}` always — `PlainTextParser` outputs empty metadata dict, so `stm_chunks.metadata` is `{}` for every chunk. The `memory_class` scoring in recall therefore always defaults to `raw_evidence`. **Score shaping is silently disabled for all MCP-ingested text.** (BUG-04)
- C. `rel_extractor` is instantiated and passed in `stdio_server.py` when `OPENROUTER_API_KEY` is set, but `MemoryIngestor._ingest_parsed()` never calls it. REL edges are never written to STM. (BUG-05)
- D. The `"Content\n\nLTM"` entity exists in Neo4j — a multi-line entity name, evidence that the `## Content` prefix stripping in ingestor has a regex edge case where strip doesn't clean multi-segment text before a session flush.

**DB evidence:**
```
stm_chunks metadata: {}          (all rows)
ltm_chunks metadata: {}          (all rows)
NEO4J dirty entity: {'e.name': 'Content\n\nLTM', 'e.type': 'POLICY'}
```

---

### 2. `memory_recall`

**Status:** ⚠️ Degraded — returns hits, but `entities: []` for ANN-only hits

**Findings:**
- A. `entity_names` on `RecallResult` is only populated by `_entity_boost()` — the Memgraph entity path. If the entity boost path doesn't run (no Memgraph ENTITY nodes for the session, or entity query returns 0), the chunk hits from the pgvector ANN lane have `entity_names = []`. For session `review-001`, Memgraph has 0 ENTITY nodes despite 15 being reported at ingest — they were flushed to Neo4j and deleted from STM. So STM recall always returns `entities: []` post-flush.
- B. `cosine_score` is now properly populated (fixed today) and visible in output.
- C. `scope="both"` normalisation works correctly: STM gets ×0.6, LTM ×0.4 weight.
- D. `_clean()` called on `r.text` in `_fmt()` strips the `## Content` prefix from display output. Correct.

**DB evidence:**
```
memory_recall hit: {"entities": [], "cosine_score": 0.446014}
Memgraph ENTITY nodes in review-001: 0  (flushed to Neo4j, not re-read at recall)
```

**Root cause of A:** `_entity_boost` queries Memgraph for live STM entities. After flush, STM is cleared. Entity names should also come from pgvector `metadata.entity_names` if stored, but they are not stored (see BUG-04).

---

### 3. `memory_flush`

**Status:** ⚠️ Degraded — works, but slow (48s) and topics never created

**Findings:**
- A. **48 second flush** for a single-chunk session. The LLM summarization path via `_summarize_topics()` calls OpenRouter; this is the dominant cost. Even when `run_gds=False`, the flush still runs community detection logic. The `memory_flush` MCP tool hardcodes `run_gds=False` but `summarize_topics` defaults to `True` in `FlushOptions` — so LLM summarization still runs.
- B. **`ltm_topics = 0` always.** Despite `summarize_topics=True`, no topics are created. Root cause: Louvain community detection requires Neo4j GDS and a minimum entity-entity edge count. With REL edges = 1 in the graph, the community algorithm produces 0 communities → 0 topics.
- C. **Flush correctly promotes** SESSION→FILE→CHUNK→ENTITY chain from Memgraph STM to Neo4j LTM. Neo4j confirms: `review-001` → FILE `49736df32f2e4de0` → CHUNK → 14 ENTITY nodes, all correct.
- D. **CHUNK nodes have no `session_id` property in Neo4j** (confirmed: `c.session_id = None` on all CHUNK nodes). This means `recall_by_document` cannot return `session_id` in provenance — as fixed today.

**DB evidence:**
```
memory_flush elapsed: 48.309s  (single chunk, 1 session)
ltm_topics count: 0
Neo4j REL edges total: 1
```

---

### 4. `memory_stats`

**Status:** ✅ Working correctly

**Findings:**
- Counts match live DB: stm_chunks=1, ltm_chunks=11 (after flush), topics=0, entities=32 (36 in Neo4j, slight lag due to merge dedup).
- Memgraph SESSION count = 11 correct.
- Topic count = 0 correct.
- Minor: STM chunk count shows 1 (the leftover stale chunk in `test-rel002-investigate` session, not the `review-001` chunk just flushed). Shows stats are live.

---

### 5. `memory_doctor`

**Status:** ✅ Working correctly

**Findings:**
- Returns `{memgraph: true, neo4j: true, pgvector: true, version: "35911e6", pid: 428872}`.
- Version field correctly reads git SHA. Clean.

---

### 6. `ltm_entity`

**Status:** ✅ Fixed today — now works case-insensitively with correct label

**Findings (fixed):**
- Was: `MATCH (e:Entity {name: $name})` — wrong label (`:Entity` vs `:ENTITY`) AND case-sensitive.
- Now: `MATCH (e:ENTITY) WHERE toLower(e.name) = toLower($name)` — correct label AND case-insensitive.
- Verified: `ltm_entity("GraphKnows")` returns `{name: "graphknows", type: "ORGANIZATION", confidence: 0.667}`.
- Remaining gap: `segment_count` always returns 0. The `OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:Segment)` clause matches against old-format `:Segment` nodes which no longer exist in the current schema (current schema uses `:SESSION`/`:CHUNK`). This count is always 0 and misleading.

**DB evidence:**
```
ltm_entity("GraphKnows") → {name: "graphknows", type: "ORGANIZATION", confidence: 0.667, segment_count: 0}
ltm_entity("GLiNER2")    → {name: "GLiNER2", type: "PRODUCT", confidence: 0.680, segment_count: 0}
```

---

### 7. `ltm_query`

**Status:** ❌ Broken by dependency — always returns `[]`

**Findings:**
- Tool implementation is correct: embeds query, does ANN on `ltm_topics.embedding`.
- Root cause: `ltm_topics` has 0 rows. Topic summarization (Louvain + LLM) requires REL edges for community detection. With ≤1 REL edge in Neo4j, no communities form, no topics written.
- **Tool is structurally correct but permanently returns empty** until the entity relation pipeline (BUG-05) is fixed and produces meaningful ENTITY→REL→ENTITY graphs.

**DB evidence:**
```sql
SELECT count(*) FROM ltm_topics;  -- 0
```

---

### 8. `ltm_topics`

**Status:** ❌ Broken by dependency — always returns `[]`

**Findings:** Same root cause as `ltm_query`. No Topic nodes in Neo4j, no rows in `ltm_topics` pgvector table.

---

### 9. `recall_by_document`

**Status:** ⚠️ Partially working

**Findings:**
- Returns correct text and entity overlap scores. Top hit for "GraphKnows memory service" correctly returns the review-001 chunk with score 2.0 (matched 2 entities).
- `provenance.session_id` was null (fixed today: CHUNK nodes have no `session_id` property, now traversed via FILE).
- `provenance.file_id` now correctly returned via `(FILE)-[:HAS_CHUNK]->(CHUNK)` traversal.
- Entity matching is keyword/substring based — not semantic. `"short-term graph memory"` is returned as an entity name match, which is a multi-word span, not a true entity. GLiNER2 extracts these long noun phrases as CONCEPT entities; the Cypher `toLower(e.name) CONTAINS token` then matches single tokens inside these spans, which may over-match.

**DB evidence:**
```
recall_by_document("GraphKnows memory service") →
  text: "GraphKnows is a universal memory service..."
  entities: ["GraphKnows", "short-term graph memory"]
  provenance: {file_id: "49736df32f2e4de0"}
  score: 2.0
```

---

### 10. `recall_by_dialog`

**Status:** ❌ Broken — always returns `[]`

**Findings:**
- Tool queries Memgraph for `TURN` nodes. Memgraph has **0 TURN nodes** in the live DB.
- `DialogWriter` is never called from the MCP `memory_ingest` path — only from `ingest_text` via direct `DialogWriter` usage. MCP `memory_ingest` uses `MemoryIngestor.ingest_text()` → no dialog structure created.
- TURNs are only written when callers explicitly use `DialogWriter.write_turn()`. The MCP server has no `write_turn` tool.
- **This tool is structurally correct but has no data to return** because the MCP surface doesn't expose dialog ingestion.

**DB evidence:**
```
Memgraph TURN nodes: 0
recall_by_dialog("entity extraction session flush") → []
```

**Fix needed:** Either add a `dialog_ingest` MCP tool that writes TURN nodes, or document that `recall_by_dialog` requires out-of-band dialog writing.

---

### 11. `recall_by_topic`

**Status:** ❌ Broken by dependency — always returns `[]`

**Findings:**
- Queries Neo4j for `Topic` nodes. Neo4j has 0 Topic nodes (same root cause as `ltm_query`/`ltm_topics`).
- Tool implementation is correct.

---

### 12. `corpus_ingest`

**Status:** ❌ Broken — always times out at MCP client 60s timeout

**Findings:**
- The graphgen pipeline (NER → RE → graph cleaning → community detection → Neo4j write-back) takes 2–10+ minutes for even a single document. The FastMCP client applies a 60s tool timeout.
- Every `corpus_ingest` call will time out. The pipeline may continue running in the background but the MCP response is lost.
- No async job ID / callback mechanism exists. The tool is synchronous-only.
- The tool docs say "GDS enrichment (Node2Vec + Louvain) runs automatically" — this is also true, adding further time.
- **This tool cannot work as a synchronous MCP call.**

**Fix:** Either (a) return a job ID and add a `corpus_ingest_status(job_id)` tool, or (b) strip GDS enrichment from the inline call and run it asynchronously, or (c) document that `corpus_ingest` must be called via CLI (`graphknows-cli graphgen run`) not via MCP.

---

## Bug Summary Table

| # | Tool | Severity | Description | Root Cause | Fix Location |
|---|------|----------|-------------|------------|--------------|
| B-01 | `memory_recall` | MEDIUM | `entities: []` on ANN-only hits post-flush | Entity names not stored in pgvector metadata; Memgraph STM cleared after flush | `ingestor.py`: store entity names in `stm_chunks.metadata`; `recall_stm.py`: read them back |
| B-02 | `memory_ingest` | MEDIUM | `metadata: {}` always — score shaping disabled | `PlainTextParser` outputs empty `seg.metadata`; no `atom_type`/`memory_class` set | `parsers/text.py`: add `atom_type: "text"` to seg metadata; `ingestor.py`: set `memory_class` from metadata (BUG-04) |
| B-03 | `memory_ingest` | MEDIUM | REL edges never written | `rel_extractor` passed but never called in `_ingest_parsed()` | `ingestor.py` (BUG-05) |
| B-04 | `ltm_entity` | LOW | `segment_count` always 0 | `OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:Segment)` — `:Segment` label no longer exists | `stdio_server.py`: change to `:CHUNK` via HAS_ENTITY, or remove the count |
| B-05 | `ltm_query` / `ltm_topics` / `recall_by_topic` | HIGH | Always empty | No topics created because no REL edges for Louvain community detection | B-03 must be fixed first; then flush with `run_gds=True` |
| B-06 | `recall_by_dialog` | HIGH | Always empty | No TURN nodes — dialog ingestion not exposed in MCP surface | Add `dialog_ingest` MCP tool or document the gap |
| B-07 | `corpus_ingest` | CRITICAL | Always times out | Synchronous pipeline takes 2–10min; MCP client timeout is 60s | Make async with job ID, or move to CLI only |
| B-08 | `memory_flush` | LOW | 48s per flush | LLM summarization via OpenRouter even for tiny sessions | Make `summarize_topics` conditional on min entity count; skip when REL edges = 0 |
| B-09 | `memory_ingest` | LOW | `"Content\n\nLTM"` dirty entity | Multi-line entity span from incomplete `## Content` stripping in edge case | `ingestor.py`: tighten regex; add `.strip()` before extraction |

---

## Bugs NOT Previously Known

- **B-04**: `ltm_entity.segment_count` always 0 — `:Segment` label is stale schema; `APPEARED_IN` relation has no targets.
- **B-06**: `recall_by_dialog` structurally broken because MCP surface has no dialog write path. Zero TURN nodes will ever exist from MCP-only usage.
- **B-07**: `corpus_ingest` never completes within the 60s MCP timeout — confirmed by live timeout. This is not a latency issue; it's a fundamental mismatch between pipeline duration and synchronous MCP call semantics.
- **B-08**: `memory_flush` slowness is dominated by LLM summarization (OpenRouter call) even when there are 0 REL edges, making community detection a no-op. The LLM call should be gated on `entity_count >= min_community_size`.
- **B-09**: Dirty multi-line entity `"Content\n\nLTM"` present in Neo4j — evidence of a regex edge case in the `## Content` strip logic.

---

## Priority Order for Fixes

1. **BUG-05** (wire `_rel_extractor`) — unblocks topics, `ltm_query`, `recall_by_topic`
2. **BUG-04** (chunk metadata) — unblocks `entities:[]` in recall, enables score shaping
3. **B-04** (segment_count fix) — 1-line Cypher change
4. **B-06** (recall_by_dialog) — needs design decision: add `dialog_ingest` tool or document as CLI-only
5. **B-07** (corpus_ingest async) — architectural change needed
6. **B-08** (flush speed) — gate LLM summarization on REL edge count > 0
