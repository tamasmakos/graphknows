# Universal Agent Memory Service — Architecture Discovery

Date: 2026-05-24
Status: Discovery complete, ready for planning

## 1. Product Pivot

GraphKnows becomes a **pure memory/knowledge service** — domain-free, schema-free, framework-agnostic.

- No EU connectors
- No EPRS report generation
- No LangGraph investigation agent
- No domain-specific pipelines
- No graph visualization

The value proposition: **Any MCP-speaking agent can plug into GraphKnows and get persistent, queryable memory.**

## 2. Integration Protocol: MCP

MCP is the universal integration layer. Every agent framework that supports MCP (Hermes, Claude Desktop, Cursor, Zed, and via tool wrappers: OpenCode, LangChain agents, CrewAI agents) can consume GraphKnows memory with zero framework-specific code.

Current MCP surface (two servers):

### graphknows_memory (stdio, PID 158299, already wired to Hermes)
- memory_doctor — backend connectivity check
- memory_stats — STM/LTM counts
- memory_ingest — text → STM (GLiNER2 entities + pgvector embedding)
- memory_recall — hybrid RRF search (STM, LTM, both)
- memory_flush — STM session → Neo4j LTM promotion

### graphknows_ltm (mcp_ltm.py, NOT wired to Hermes)
- ltm_query — semantic topic search (pgvector)
- ltm_entity — named entity lookup (Neo4j)
- ltm_topics — topic listing by threshold
- recall_by_document — document chunks by entity match
- recall_by_dialog — conversation utterances by entity match
- recall_by_topic — topic matching

## 3. Prior Research (Already Validated)

### Generic Agent Adapter Contract
File: `research/generic_agent_contract_draft.md`

Defines:
- `AgentMemoryAdapter(Protocol)` — open_session, ingest_artifact, write_turn, recall, promote, forget
- Typed records: ArtifactRecord, TurnRecord, ToolCallRecord, RecallRequest, RecallHit, PromoteOptions
- Write receipts, idempotency semantics, error policies
- Evidence vs memory distinction (raw trace stored separately from distilled memory)

### Feasibility Prototype (v1)
File: `research/proto_generic_memory_feasibility_report.md`

Validated on real Hermes session (127 messages):
- Raw events can be parsed from Hermes SQLite without framework changes
- Distilled memory (State/Episode/Fact) beats raw trace on recall quality
- Key finding: raw terminal path spam → useless recall; distilled summaries → useful recall
- Issues: flat recall scores (~0.015-0.016), GLiNER2 terminal noise, `## Content` wrapper pollution

### Deep Prototype (v2)
File: `research/proto_generic_memory_v2.py` + `research/proto_generic_memory_v2_results.json`

Validated on 3 real Hermes sessions (24+127+208 messages, 359 total):
- Multi-session parsing working
- Step segmentation (assistant message + following tool messages)
- Automatic State/Episode/Fact distillation (LLM-assisted, not hand-authored)
- Cross-session promotion (3-8 promoted facts derived from pattern overlap)
- A/B lanes: raw vs step-summary vs distilled
- Quantitative quality signals: noise entity rate, useful-hit rate, promoted-fact coverage

## 4. Architecture: What Stays vs What Goes

### STAYS (Core Memory Engine)

```
graphknows/
  memory/
    ingestor.py          — MemoryIngestor (text → STM)
    extractor.py         — GLiNER2EntityExtractor
    dialog.py            — DialogWriter (turn ingestion)
    flush.py             — SessionFlusher (STM → LTM)
    adapter.py           — MemoryAdapter (host-agnostic interface)
    adapter_types.py     — Typed records
    tool_summarizer.py   — Tool output extraction
  retrieval/
    stores/              — MemgraphStore, PgVectorStore, Neo4jStore
    recall_stm.py        — Short-term recall
    recall_ltm.py        — Long-term recall
    stores/embedder.py   — Sentence transformer embeddings
    retriever.py         — Retrieval pipeline
  mcp/
    stdio_server.py      — Primary MCP server (5 tools)
  mcp_ltm.py             — LTM MCP server (6 tools)
  feature_engineering.py — GDS enrichment (Louvain, PageRank, betweenness, Node2Vec)
  forget.py              — Memory pruning
  schema.py              — Graph schema (SESSION, FILE, CHUNK, ENTITY, DIALOG, TURN)
  ingestion/parsers/     — Text, code, docx parsers
  ontology/
    rel_extractor.py     — DSPy OntologyRelExtractor (relation extraction)
    loader.py            — Ontology loading
  cli/memory.py          — CLI (graphknows-memory)
  migrations/            — STM/LTM schema migrations
  container.py           — DI container
  config.py              — Settings
```

### GOES (Domain-Specific & Agent Pipeline)

```
graphknows/graphgen/                  — DSPy graph generation pipeline (domain-specific)
graphknows/api.py                     — FastAPI REST server (investigation endpoint)
graphknows/run.py                     — Agent runner
graphknows/ontology/router.py         — Domain keyword routing
scripts/build_libe_brief.py           — EU report generation
research/prototype_archive_*.py       — EU archive prototypes
research/run_investigation.py         — Agent investigation runner
research/agent_demo.py                — Agent demo
research/demo_*.py                    — Domain demos
research/entity_kg.py                 — Domain KG
apps/web/                             — Web frontend
output/                               — Generated reports
Dockerfile                            — Agent service (replace with memory-only Dockerfile)
```

## 5. Gaps — Missing Universal Tools

| Tool | Status | Priority |
|------|--------|----------|
| memory_delete / forget_session | Protocol defined, not in MCP | High |
| memory_search_entities | Not implemented | High |
| memory_get_graph (subgraph around entity) | Not implemented | Medium |
| memory_list_sessions | Not implemented | Medium |
| memory_export_session | Not implemented | Low |
| memory_summarize (on-demand topic summarization) | Exists in CLI only | Medium |

## 6. Consolidation Plan

### Phase 1: Unify MCP Servers
Merge `graphknows_memory` + `graphknows_ltm` → single `graphknows` MCP server.

All tools under one namespace:
- `memory_*` (STM operations)
- `ltm_*` (LTM queries)
- `recall_*` (cross-scope)
- `admin_*` (doctor, stats)

### Phase 2: Implement AgentMemoryAdapter Protocol
Build the `AgentMemoryAdapter` class defined in the contract draft. This is the in-process Python API that backs the MCP server.

### Phase 3: OpenCode Integration
Ship `.opencode/tools/graphknows-recall.ts` and `graphknows-ingest.ts` — thin TypeScript wrappers that shell out to the MCP server via stdin JSON-RPC (same pattern as existing `db_debug.py` bridge).

### Phase 4: Documentation
Integration guide covering:
- Wiring GraphKnows MCP into Hermes (config.yaml entry)
- Wiring GraphKnows MCP into OpenCode (tool wrappers)
- Wiring into any MCP-speaking agent (Claude Desktop, Cursor, Zed)
- Session lifecycle: open → ingest → recall → flush → forget

## 7. Hermes-Specific Integration Notes

Hermes already has `graphknows_memory` MCP wired. To add LTM tools:
- Add `graphknows_ltm` MCP server entry to Hermes config (or use unified server after Phase 1)

Hermes session auto-ingest pattern (prototyped in `proto_hermes_session_ingest.py`):
- On session end: parse Hermes SQLite session → distill State/Episode/Fact → ingest into GraphKnows STM → flush to LTM
- Plugin hook points: `initialize_all` (register provider), `tool_complete_callback` (async STM write), `session_end` (LTM flush)

## 8. OpenCode-Specific Integration Notes

OpenCode does not natively speak MCP. Integration pattern:
1. Add `.opencode/tools/graphknows-*.ts` tools
2. Each tool shells out to the MCP server via stdin JSON-RPC probe
3. Pattern: `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call",...}' | graphknows-mcp`
4. This is the same bridge pattern as existing `memgraph-query.ts` → `db_debug.py`

Tools to add:
- `graphknows-recall` — recall from memory (STM/LTM/both)
- `graphknows-ingest` — ingest text/file into memory
- `graphknows-entity` — lookup entity in knowledge graph
- `graphknows-flush` — promote session to LTM

## 9. Key Decisions Made

1. **MCP is the universal protocol.** No framework-specific integrations. Hermes and OpenCode are just clients.
2. **Two write paths, two recall paths, one promotion path.** The generic contract stands.
3. **Distilled > raw for recall.** Validated. Raw traces are evidence; distilled objects are memory.
4. **Domain-free means removing graphgen, EU connectors, agent pipeline.** Clean break.
5. **Memory-level schema stays.** SESSION→FILE→CHUNK→ENTITY, DIALOG→TURN — these are domain-agnostic.

## 10. Related Files

- Contract draft: `research/generic_agent_contract_draft.md`
- Feasibility v1: `research/proto_generic_memory_feasibility_report.md`
- Feasibility v2 results: `research/proto_generic_memory_v2_results.json`
- Hermes session ingest: `research/proto_hermes_session_ingest.py`
- OpenCode brief v2: `research/opencode_feasibility_brief_v2.md`
- Current MCP server: `graphknows/mcp/stdio_server.py`
- LTM MCP server: `graphknows/mcp_ltm.py`
- Hermes config: `/opt/data/config.yaml` (MCP servers section)
- OpenCode tools: `/root/graphknows/.opencode/tools/` (existing DB tools)