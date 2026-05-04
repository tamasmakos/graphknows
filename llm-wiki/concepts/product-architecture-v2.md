---
title: Product Architecture Redesign — Graphknows v2
created: 2026-05-04
updated: 2026-05-04
type: concept
tags: [architecture, product, planning, graphrag, graphgen, agent, search]
sources: []
confidence: high
---

# Product Architecture Redesign — Graphknows v2

## Vision

A political intelligence platform for **policy researchers, politicians, journalists and oversight professionals** who need to reason over large volumes of public records from nations and public organizations — and produce exportable, shareable visualizations to argue and convince.

**Primary differentiator**: Ask in plain English → grounded answer with source citations → live knowledge graph visualization → one-click export (PNG, SVG, interactive HTML, shareable link).

---

## Core Design Decisions

### 1. Only public records — no noise
- Parliamentary debates, legislation, regulations, EU directives, UN resolutions, court decisions, government reports
- No news, social media, or private documents
- One data source type per country: the official parliamentary/legislative record

### 2. Agent-first architecture
- Frontend interacts only with the **Agent service**
- The Agent builds a **per-session context graph** (session overlay on top of shared KG)
- Agent orchestrates: search → scrape → ingest → reason → visualize
- GraphGen and GraphRAG become powerful tools the agent calls, not end-user products

### 3. Service decomposition
Old layout:
```
scrapers/ (flat folder)
services/graphgen/
services/graphrag/ (contains the agent)
apps/web/
```

New layout:
```
services/
  search/          ← NEW: was scrapers/ — sophisticated agentic document search
  graphgen/        ← enhanced: expert KG construction tool
  graphrag/        ← enhanced: expert retrieval tool (no agent inside)
  agent/           ← NEW: orchestrator — the only entry point from frontend
apps/web/          ← frontend: chat + live graph viz + export
```

### 4. Session-scoped context graph
- **Persistent shared KG (Neo4j)** = background corpus — all ingested political documents
- **Per-session ephemeral overlay (Memgraph)** = session context — in-memory, sub-ms latency
- At session start: agent seeds Memgraph with a relevant subgraph extracted from Neo4j
- Agent accumulates entities/relationships discovered during the session into Memgraph
- Memgraph runs `IN_MEMORY_ANALYTICAL` mode for fast bulk load, then `IN_MEMORY_TRANSACTIONAL` for ACID session writes
- Session graph is visualized live in the frontend as the agent reasons (WebSocket stream from Memgraph)
- MAGE algorithms (Leiden/Louvain, PageRank, BFS/DFS, Betweenness Centrality) run natively in Memgraph — no extra license
- On session end: user can export the session graph; validated inferences can be written back to Neo4j
- Isolation: one Memgraph container per session (Community Edition) or multi-tenancy (Enterprise)
- Both databases share Bolt protocol and Cypher syntax — same Python driver (`neo4j` package) for both

**Why Memgraph over Redis or in-memory Neo4j:**
- Native in-memory C++ engine: sub-millisecond traversals vs JVM/disk overhead of Neo4j
- MAGE graph algorithms bundled (PageRank, community detection, influence scoring)
- Full Bolt + openCypher compatibility — minimal code change from Neo4j queries
- Per-container memory limits (`--memory-limit`) bound session RAM usage cleanly
- `MATCH (n) DETACH DELETE n` or container restart = trivial session teardown

**Docker:**
```bash
docker run -p 7688:7687 -p 7445:7444 --name memgraph-session memgraph/memgraph-mage
```
(Use port 7688 locally to avoid collision with Neo4j on 7687)

### 5. Visualization as first-class export
- Live graph canvas (D3.js force-directed) synced to agent reasoning
- Exports: PNG (print-quality), SVG (Illustrator-friendly), self-contained HTML (offline), shareable permalink
- Argument cards (social-format single-relationship highlight)
- PDF investigation reports (auto-generated narrative)

---

## New Service Architecture

### `services/search/` (was `scrapers/`)

**Purpose**: Agentic document discovery and ingestion from public official sources.

**Responsibilities**:
- Country-specific scrapers (each a well-maintained adapter)
- Agentic search: given a topic or entity, find relevant documents across sources
- Document retrieval, parsing, format normalization (PDF, XML, HTML, Akoma Ntoso)
- Deduplication, freshness tracking, incremental ingest
- Expose a REST API: `POST /search`, `POST /ingest`, `GET /sources`, `GET /status`

**Key sources** (per country):
- EU: EP Open Data API, EUR-Lex OData (metadata), EUR-Lex full text (where available)
- UK: legislation.gov.uk Atom feed, Hansard API, TheyWorkForYou
- Germany: Bundestag Open Data API (DIP)
- France: Assemblée Nationale open data
- International: UN Document System, OHCHR, WTO DDF

**Agentic search capability** (new):
- ReAct agent that decomposes a user topic into search queries across multiple sources
- Follows document references (cross-citations between bills, treaties)
- Returns ranked document list with relevance scores

### `services/graphgen/` (enhanced)

**Purpose**: Expert KG construction from text. Called by agent, not by users.

**API**: `POST /build` → accepts document chunks, returns KG (entities + relations)
**Key enhancements needed**:
- Fix label collapse (Task T2)
- Fix community_id propagation (Task T3)
- Typed LKIF relation edges via DSPy
- Temporal entity linking (same entity across time periods)
- Political entity classification improvements

### `services/graphrag/` (enhanced, no agent inside)

**Purpose**: Expert hybrid retrieval over the shared KG. Called by agent, not by users.

**API**: `POST /retrieve` → accepts query + session_id, returns ranked chunks + subgraph
**Key enhancements needed**:
- pgvector retrieval verification (Task T1)
- Full corpus ingest (Task T4)
- `get_chunks_for_entity` implementation (Task T5)
- RAGAS eval (Task T6)

### `services/agent/` (new — extracted from graphrag)

**Purpose**: The intelligent orchestrator. The only service the frontend talks to.

**Responsibilities**:
- Manage chat sessions
- Build and maintain per-session context graph (session overlay on shared KG)
- Orchestrate: search → graphgen → graphrag → visualize loop
- Expose session graph state to frontend via WebSocket (live updates)
- Produce citations and reasoning traces for every answer
- Tool set: search_documents, build_kg, retrieve_context, traverse_graph, synthesize_answer

**Session graph technology: Memgraph**
- One Memgraph instance per session (Docker container, `memgraph/memgraph-mage`)
- Port: 7688 (avoids collision with persistent Neo4j on 7687)
- Python client: `neo4j` driver (Bolt compatible) — same driver used for Neo4j
- MAGE algorithms run inside Memgraph for session-level analytics (centrality, community detection, influence path)
- Session teardown: `MATCH (n) DETACH DELETE n` or container kill

**Session graph model (Memgraph nodes/edges):**
```cypher
-- Seeded from Neo4j at session start
MERGE (e:Entity {name: $name}) SET e.label=$label, e.source_session=$session_id
MERGE (a:Entity {name: $a})-[r:RELATED_TO {type: $rel_type, source: $doc_id, weight: $w}]->(b:Entity {name: $b})

-- Reasoning trace nodes (session-only)
MERGE (s:ReasoningStep {id: $step_id})
SET s.tool=$tool, s.query=$query, s.result_summary=$summary, s.timestamp=$ts
MERGE (s)-[:RETRIEVED]->(e:Entity {name: $entity_name})
```

**Write-back to Neo4j (on session close, user-triggered):**
```python
# Export confirmed edges from Memgraph → write to Neo4j
# Only write edges with confidence > threshold and user approval
```

### `apps/web/` (new frontend design)

**Core UX**: Split-pane — chat (35%) + live graph canvas (65%)

**Key features**:
- Chat drives graph: agent claims highlight graph subgraph in real time
- Graph drives chat: click node → "Tell me more" pre-populates
- Investigation sessions (named, saved, resumable)
- Live reasoning trace sidebar (collapsible DAG of agent steps)
- One-click exports: PNG, SVG, HTML, shareable link, PDF report
- Argument cards: select a relationship → share card generator

---

## Research Validation

### Market confirms the gap
- No competitor combines: NL query + graph viz + provenance + shareable export
- Key competitors (FiscalNote, Quorum, Capitol.ai) are keyword/CRM-based — no real semantic graph
- Beachhead: investigative journalists + policy researchers (fastest adoption, highest word-of-mouth)

### Academic evidence supports approach
- Hybrid (dense+graph) retrieval: +8–15% F1 over pure RAG on legal/political multi-hop queries (LegalBench-RAG 2407.11011, LEGALM 2411.11108)
- Session-scoped overlay > full per-session KG construction: 60–70% lower latency, comparable accuracy (2404.11893)
- ParlGraph [2404.17741], HiRAG [2411.09011] validate the parliamentary KG approach
- Akoma Ntoso XML is the standard for parliamentary data interoperability
- GraphStory [2311.04291] validates chat-driven narrative generation from KGs for journalism

### UX principles validated
- Investigation-scoped sessions (not ephemeral chats) — Palantir/Linkurious model
- Claim-to-subgraph highlighting is the key differentiator — no competitor does this
- Self-contained offline HTML export is critical for journalist secure environments
- Provenance per node is a trust signal that drives enterprise adoption

---

## Kanban Task Graph (v2 Redesign)

See [[kanban-plan]] for full board.

Phases:
1. **Research & Architecture** (this doc) — DONE
2. **Phase 1 baseline fixes** (T1–T6) — prerequisite for everything
3. **Service extraction** (T7–T10) — extract agent, create search service
4. **Enhanced graphgen** (T11–T13) — relations, temporal, political entity fixes
5. **Frontend v2** (T14–T16) — split pane, live graph, exports
6. **Multi-country expansion** (T17+) — Bundestag, Assemblée, UN

---

## Links

[[business-plan]] [[market-analysis]] [[graphrag-and-downstream]] [[graphgen-architecture]] [[project-roadmap]] [[kanban-plan]]
