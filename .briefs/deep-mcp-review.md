# Deep MCP Quality Review — All 12 Tools

## Context

Working directory: /root/graphknows  
Branch: main @ 35911e6  
MCP binary: /root/graphknows/.venv/bin/graphknows-mcp  
All 3 backends are live: Memgraph bolt://172.18.0.1:17687 (no auth), Neo4j bolt://172.18.0.1:7687, pgvector 172.18.0.1:5432/vectordb

You have database tools available:
- `neo4j-query` tool: query Neo4j via Cypher
- `memgraph-query` tool: query Memgraph via Cypher  
- `pgvector-query` tool: query pgvector via SQL

## Task

Do a thorough quality audit of the 12 MCP tools defined in `graphknows/mcp/stdio_server.py`.
For each tool: read the implementation, query the live DBs, identify real bugs, schema mismatches, missing data, and missing error handling.

## The 12 Tools to Review

**STM (5):**
1. `memory_ingest` — MemoryIngestor → Memgraph + pgvector stm_*
2. `memory_recall` — RRF recall from stm_chunks + stm_turns (scope: stm/ltm/both)
3. `memory_flush` — SessionFlusher STM → LTM promotion
4. `memory_stats` — counts from Memgraph + Neo4j + pgvector
5. `memory_doctor` — backend health check

**LTM (6):**
6. `ltm_entity` — entity lookup by name in Neo4j ENTITY nodes
7. `ltm_query` — semantic search on ltm_topics via pgvector embedding
8. `ltm_topics` — list Topic nodes above entity_count threshold
9. `recall_by_document` — keyword entity match → CHUNK nodes in Neo4j
10. `recall_by_dialog` — keyword entity match → TURN nodes in Memgraph
11. `recall_by_topic` — find topics by label/entity match

**Corpus (1):**
12. `corpus_ingest` — batch graphgen pipeline → Neo4j

## Known Issues (Principal Engineer found these already — verify and find MORE)

1. **`entities: []` in memory_recall hits** — STM recall returns hits with `entities: []` even when 15 entities were extracted at ingest. The entity names should be populated from the pgvector metadata or Memgraph. Root cause unclear — trace through `recall_short_term()` in `graphknows/retrieval/recall_stm.py`.

2. **`ltm_query` returns `[]` always** — `ltm_query` does semantic search on `ltm_topics` via pgvector embedding. But `ltm_topics` has 0 rows (no topics populated because `memory_flush` is called with `run_gds=False` which skips topic summarization). The tool works for its stated purpose, but is useless without topics. Check if `run_gds=False` is the right default, or if topics should be populated differently.

3. **`ltm_topics` always returns `[]`** — same root cause as above. No topics exist.

4. **`recall_by_dialog` returns `[]`** — queries TURN nodes in Memgraph. Check if TURN nodes actually exist via `memgraph-query`.

5. **`recall_by_topic` returns `[]`** — no topics in Neo4j Topic nodes either. Verify with `neo4j-query`.

6. **`corpus_ingest` timed out at 60s** — the MCP client has a 60s timeout but the corpus pipeline is much slower. This means corpus_ingest ALWAYS times out for any non-trivial input. Check if it can be made async or if the MCP timeout needs to be configurable.

7. **`memory_flush` takes 48s** — extremely slow for a single-chunk session. Profile what takes so long (likely the DSPy LLM call via OpenRouter). Is there a fallback when DSPy/OpenRouter is unavailable?

## Investigation Steps

For each tool:
1. Read the implementation in `stdio_server.py` (find the tool handler)
2. Read the underlying implementation file it calls
3. Query the live DB to see actual data
4. Identify: crashes, wrong SQL/Cypher, missing data, timeout risks, bad error messages

### Key files to read:
- `graphknows/mcp/stdio_server.py` — all 12 tool handlers
- `graphknows/retrieval/recall_stm.py` — memory_recall implementation
- `graphknows/retrieval/recall_ltm.py` — LTM recall
- `graphknows/recall.py` — recall_by_document/dialog/topic
- `graphknows/memory/flush.py` — memory_flush
- `graphknows/memory/adapter.py` — memory_stats

### Key DB queries to run:

```sql
-- pgvector: check stm_chunks entity data
SELECT id, session_id, array_length(string_to_array(metadata::text, ','), 1), metadata 
FROM stm_chunks LIMIT 5;

-- pgvector: check ltm_topics
SELECT * FROM ltm_topics LIMIT 5;

-- pgvector: check ltm_chunks
SELECT id, session_id, metadata FROM ltm_chunks LIMIT 5;
```

```cypher
-- Memgraph: check what node types exist
MATCH (n) RETURN labels(n), count(n) LIMIT 20;

-- Memgraph: check TURN nodes
MATCH (t:TURN) RETURN t LIMIT 5;

-- Memgraph: check ENTITY nodes with session_id
MATCH (e:ENTITY) RETURN e.name, e.session_id LIMIT 10;
```

```cypher
-- Neo4j: check Topic nodes
MATCH (t:Topic) RETURN t LIMIT 5;

-- Neo4j: check ENTITY nodes
MATCH (e:ENTITY) RETURN e.name, e.type LIMIT 10;

-- Neo4j: check what labels exist
CALL db.labels() YIELD label RETURN label;
```

## Output Format

Write a report to `/root/graphknows/llm-wiki/ops/mcp-deep-review-20260525.md` with:

```markdown
# MCP Deep Review — 2026-05-25

## Tool-by-Tool Findings

### 1. memory_ingest
- Status: ✅ Working / ⚠️ Degraded / ❌ Broken
- Bugs found: ...
- DB evidence: ...

[repeat for all 12]

## Bug Summary Table

| Tool | Severity | Description | Root Cause | Fix Location |
|------|----------|-------------|------------|--------------|

## Bugs NOT Previously Known
[list only NEW findings not in the known issues above]
```

Do NOT commit anything. Do NOT modify any source files. Read and report only.

## Acceptance Criteria

- All 12 tools analyzed
- Live DB evidence for each finding
- Report written to /root/graphknows/llm-wiki/ops/mcp-deep-review-20260525.md
- `git status --short` shows only untracked wiki file (no source modifications)
