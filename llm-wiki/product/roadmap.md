---
title: "Project Roadmap — Deep Research Agent"
created: 2026-05-12
updated: 2026-05-20
type: product
tags: [roadmap, planning]
confidence: high
---

# Project Roadmap — Deep Research Agent

See [[architecture/product-architecture]] for the architecture this roadmap delivers.

---

## Done — Shipped

- **services/search/** — 13 EU + UK source connectors; job queue; idempotent ingestion; MCP server
- **services/graphknows/** — GLiNER2 NER + DSPy relation extraction; ontology-guided domain-agnostic extraction; HybridRetriever (pgvector + Neo4j/Memgraph + RRF); 3-phase patch runner
- **services/agent/** — LangGraph StateGraph (scope→investigate→communicate); WebSocket streaming; EPRS-style brief generation; session model
- **apps/web/** — Chat UI + D3 graph canvas; split-pane layout; named sessions sidebar (localStorage)
- **Memgraph integration** — per-session ephemeral session graphs; bolt driver; MERGE-based writes; pluggable backend (ADR-005)
- **pgvector integration** — schema with `session_id` column; BAAI/bge-small-en-v1.5 embeddings; session-scoped search
- **Dual-write** — `investigate_node` writes chunks to pgvector + entities/relations to Memgraph per iteration
- **Session lifecycle** — `session/manager.py`, `session/models.py`, `session/end.py`; Memgraph + pgvector session cleanup
- **Ontology system** — domain-agnostic routing (embedding-similarity → keyword → LLM fallback); legal/politics/finance/science assets
- **WriteBackService** — promotes high-confidence session entities to Neo4j corpus post-session

---

## Active — In Progress

These are the highest-priority open items as of 2026-05-20. Track in Kanban.

| # | Item | Area | Priority |
|---|---|---|---|
| 1 | **HybridRetriever Memgraph path** — entity graph traversal reads Neo4j, not Memgraph session graph. Breaks per-session retrieval correctness. | agent/retrieval | HIGH |
| 2 | **Session TTL wiring** — `session/end.py` exists but cleanup is not triggered. Memgraph memory grows unbounded. | agent/session | HIGH |
| 3 | **pgvector session_id enforcement** — audit all `PgVectorStore.search()` call sites; add integration test for session isolation | infra | HIGH |
| 4 | **End-to-end integration test** — one real investigate loop against live Memgraph + pgvector | testing | MEDIUM |
| 5 | **Embedder singleton** — move BAAI/bge-small-en-v1.5 load to module-level; remove per-call cold start | agent | MEDIUM |

---

## Next — After Active Items

- **RAGAS evaluation (per-session)** — baseline evaluation of retrieval quality on the new per-session architecture (old RAGAS results were pre-pivot and are not comparable)
- **Stub doc alerting** — WS `search_error` event + circuit breaker when search service unavailable
- **Prompt versioning** — EPRS system prompt to `graph/prompts/eprs.md`, configurable
- **Frontend export** — PDF/Markdown export of EPRS briefs
- **Source coverage expansion** — Bundestag DIP API, UK Hansard (planned, not yet implemented)
- **Dead code cleanup** — delete `services/agent/orchestrator.py` (pre-LangGraph ReAct legacy)
- **Ontology asset deduplication** — symlink `services/agent/ontology/assets/` → `services/graphknows/`

---

## Horizon — Strategic

- **Multi-language support** — French, German, Hungarian document ingestion (current pipeline is English-only)
- **Provenance graph** — PROV-O compliant lineage from source document to graph node to brief claim
- **Persistent corpus** — opt-in write-back of high-confidence session entities to shared Neo4j corpus; corpus maintenance pipeline
- **MCP surface** — expose agent as an MCP tool for third-party integrations (search MCP server already exists)
- **ELTE thesis finalisation** — Node2Vec + community detection results on EP debate corpus; formal evaluation section
