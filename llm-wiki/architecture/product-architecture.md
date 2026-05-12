---
title: "Product Architecture v2 — Deep Research Agent"
created: 2026-05-12
updated: 2026-05-12
type: concept
tags: [architecture, agent, memgraph, pgvector, langgraph]
confidence: high
---

# Product Architecture v2 — Deep Research Agent

## Vision

Graphknows is a **per-session deep research agent** for investigative journalists, policy researchers, and politicians. Given a research topic, the agent exhaustively discovers, extracts, and synthesises all relevant EU public-record and web evidence into a structured, cited, exportable EPRS-style briefing. The Knowledge Graph is a **per-session research artifact**, not a shared database to maintain or query.

The old model — pre-loading a monolithic shared KG from EU records and letting users query it — is retired. It did not scale to bespoke research needs and created unsustainable ingestion/maintenance burden.

See also: [[deep-research-agent-design]], [[dual-write-hybrid-retrieval]]

---

## Core Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| KG scope | Per-session, namespaced by `session_id` | Avoids shared-state conflicts; enables TTL cleanup |
| Graph store | Memgraph (bolt://127.0.0.1:17687) | Low-latency in-memory graph; nodes labelled with `session_id` |
| Vector store | pgvector, `session_id` column filter | Cheap dense retrieval without cross-session bleed |
| Retrieval strategy | Hybrid: pgvector dense + 2-hop Memgraph + RRF | Maximises recall/precision for structured EU text |
| Stopping criterion | Novelty signal: dup-chunk ratio > 0.6 OR iterations > 12 | Prevents runaway search without hard iteration caps alone |
| Output format | EPRS-style structured briefing | Matches target users' existing workflow expectations |
| Entry point | `services/agent/` only — frontend never calls search/graphgen directly | Clean service boundary; streaming via WebSocket |

---

## Agent Loop

```
plan_node → search_loop (novelty stopping) → dual_write (KG+pgvector) → hybrid_retrieval → brief_node
```

**Phase 1 — Scope** (`graph/nodes/scope.py`):
- `plan_node` decomposes the research topic into sub-questions and a ranked source list.
- Output: `InvestigativeState.plan` — list of query strings + source priorities.

**Phase 2 — Investigate** (`graph/nodes/investigate.py`):
- `search_loop` iterates: query → fetch → embed → novelty check.
- `dual_write` flushes each iteration's chunks to pgvector and extracted entities/relations to Memgraph, both namespaced by `session_id`.
- Loop exits when `dup_ratio > 0.6` or `iteration_count > 12`.

**Phase 3 — Communicate** (`graph/nodes/communicate.py`):
- `hybrid_retrieval` pulls top-8 evidence chunks via RRF over pgvector dense (top-20) + 2-hop Memgraph walk.
- `brief_node` renders the EPRS-style briefing with inline citations, entity summary table, and source list.

Full detail: [[deep-research-agent-design]]

---

## Service Map

```
apps/web/                         ← frontend: chat UI + D3 graph viz
  │
  └─ WebSocket / REST
       │
services/agent/                   ← ONLY entry point from frontend
  │  graph/workflow.py             ← LangGraph StateGraph (3 macro phases)
  │  graph/state.py                ← InvestigativeState schema
  │  graph/nodes/scope.py          ← plan_node
  │  graph/nodes/investigate.py    ← search_loop + dual_write
  │  graph/nodes/communicate.py    ← brief_node
  │  retrieval/hybrid_retriever.py ← pgvector dense + 2-hop Memgraph + RRF
  │  session/manager.py            ← session lifecycle, TTL cleanup
  │  tools/search_client.py        ← HTTP client → search service
  │
services/search/                  ← agentic document discovery
  │  Tavily web adapter
  │  EU adapters: EP, EUR-Lex, Council, EC, ECB, ECA, EPRS
  │
services/graphgen/                ← NER + relation extraction
     GLiNER2 entity recognition
     DSPy relation extraction
```

---

## Data Flow

```
User topic
  → plan_node          (sub-questions, source plan)
  → search_loop[n]     (HTTP → services/search → raw docs)
  → graphgen call      (NER+RE → entities, relations)
  → dual_write         (pgvector chunks | Memgraph nodes/edges, both session_id-scoped)
  → [novelty check]    → loop or exit
  → hybrid_retrieval   (pgvector top-20 → entity extract → 2-hop Memgraph → RRF → top-8)
  → brief_node         (EPRS briefing, citations, entity table)
  → WebSocket stream   → apps/web
```

Detailed write/retrieval mechanics: [[dual-write-hybrid-retrieval]]

---

## Infrastructure

| Component | Address | Role |
|---|---|---|
| Memgraph | bolt://127.0.0.1:17687 | Per-session KG; nodes labelled `session_id` |
| pgvector | postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb | Chunk store; filtered by `session_id` column |
| Neo4j | bolt://172.18.0.1:7687 | Background corpus + entity resolution only (not session KG) |
| Embed model | BAAI/bge-small-en-v1.5 | All dense embeddings |

Neo4j is **not** used for session-level KG storage. It remains available for cross-session entity canonicalization and background corpus lookups.

Session lifecycle (creation, TTL expiry, cleanup) is managed by `session/manager.py`.

---

## Non-Goals

- **Shared persistent KG** — the old monolith model is retired; no batch EU ingestion pipelines.
- **User-driven graph exploration** — the D3 viz is a research byproduct, not the product.
- **Real-time alerts / push** — out of scope for v2; horizon item.
- **Multi-user auth** — single-tenant for now; see roadmap horizon.
- **Cross-session entity merging** — Neo4j corpus used for resolution, but session KGs are not merged.
