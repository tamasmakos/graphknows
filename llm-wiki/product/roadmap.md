---
title: "Project Roadmap — Deep Research Agent"
created: 2026-05-12
updated: 2026-05-12
type: concept
tags: [roadmap, kanban, planning]
confidence: high
---

# Project Roadmap — Deep Research Agent

See [[product-architecture-v2]] for the architecture this roadmap delivers. See [[kanban-plan]] for full task registry with IDs and dependencies.

---

## ✅ Done — Shipped

- **services/search/** — Tavily + EU source adapters (EP, EUR-Lex, Council, EC, ECB, ECA, EPRS); job queue; idempotent ingestion
- **services/graphgen/** — GLiNER2 NER + DSPy relation extraction pipeline; HTTP API; ontology injection
- **services/agent/ skeleton** — LangGraph StateGraph stub; WebSocket streaming; session model
- **apps/web/ v1** — Chat UI + D3 graph visualisation; split-pane layout; reasoning trace panel
- **Memgraph integration** — bolt driver; session-namespaced node writes; CRUD helpers
- **pgvector integration** — schema with `session_id` column; BAAI/bge-small-en-v1.5 embeddings
- **RAGAS evaluation harness** — offline retrieval quality metrics baseline

---

## 🔥 Now — Active Sprint (7 Cards)

These 7 kanban cards constitute the core pivot deliverables:

| ID | Card | Owner area |
|---|---|---|
| `product-vision-rewrite` | Rewrite all wiki docs to reflect deep-research-agent model | PM/docs |
| `deep-research-loop` | Implement `search_loop` with novelty stopping in `investigate.py` | agent |
| `dual-write` | `DualWriter.write()` — flush chunks to pgvector + entities/relations to Memgraph per iteration | agent |
| `hybrid-retrieval` | `HybridRetriever.retrieve()` — pgvector dense top-20 → entity extract → 2-hop Memgraph → RRF → top-8 | agent/retrieval |
| `brief-node` | `brief_node` — EPRS-style structured briefing renderer with inline citations | agent/communicate |
| `session-lifecycle` | `session/manager.py` — session creation, state persistence, TTL expiry, Memgraph + pgvector cleanup | agent/session |
| `pgvector-namespacing` | Enforce `session_id` filter on all pgvector reads; migration if needed | infra |

Stopping criterion implementation detail: [[deep-research-agent-design]]
Write/retrieval implementation detail: [[dual-write-hybrid-retrieval]]

---

## 📋 Next

- **Session resume** — persist `InvestigativeState` to disk/Redis; allow users to reload a prior research session
- **Frontend v2** — expose live KG growth in D3 during search_loop; per-node citation panel; entity timeline
- **Export suite** — PDF export of EPRS briefing; CSV entity table; JSON citation list; Zotero-compatible RIS
- **Bundestag adapter** — extend `services/search/` with Bundestag Drucksachen + Plenarprotokolle adapters; parity with EP adapter

---

## 🔭 Horizon

- **Multi-user auth** — JWT + per-user session namespace; RBAC for shared sessions within an organisation
- **Federated KG** — merge entity nodes across sessions for returning users; cross-session entity resolution via Neo4j
- **Embed API** — expose `POST /research` as a headless API; webhook callbacks for async briefing delivery; embed widget for third-party news CMS
