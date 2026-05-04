---
title: Project Roadmap — Graphknows v2
created: 2026-05-03
updated: 2026-05-04
type: concept
tags: [roadmap, planning, architecture, product]
confidence: high
---

# Project Roadmap — Graphknows v2

> Living document. Phases are sequential gates; each phase must reach its exit criteria before the next begins. Within a phase, tracks (fix/arch/data/frontend) run in parallel.

---

## ✅ Phase 0 — Thesis Baseline (COMPLETE)

**Goal:** produce a working graph from EP debates to validate the extraction pipeline.

| Metric | Result |
|---|---|
| Corpus | 13 EP 'This is Europe' debates, 2022–2024 |
| KG size | 6,377 nodes, 20,661 edges |
| Entities | 3,557 entity nodes |
| Communities | 70 topics, 341 subtopics |
| Modularity | 0.748 → **0.823** (Node2Vec improvement) |
| Validation | Scale-free topology confirmed; EPRS thematic comparison |

**Done:** GraphML export, Neo4j adapter, pgvector schema, LDA baseline comparison, LLM summarization.

---

## 🔧 Phase 1 — Foundation Hardening (NOW — ~4 weeks)

**Goal:** fix all known bugs and technical debt so the pipeline produces production-quality KG output. Exit criterion: RAGAS faithfulness > 0.75 on held-out EP queries.

### Track A — Pipeline Bug Fixes (parallel, HIGH priority)
| Task ID | Title | Est. |
|---|---|---|
| t_66aa8b14 | fix: pgvector retrieval end-to-end (integration test) | 2d |
| t_930b0b51 | fix: entity label collapse — induced label set, no UNKNOWN fallback | 3d |
| t_d5f42600 | fix: community_id null on Topic nodes — trace detection.py + neo4j_adapter | 2d |
| t_40558195 | feat: NeoStore.get_chunks_for_entity() implementation | 2d |
| t_b2627d7f | fix: write typed LKIF_REL edges to Neo4j from DSPy output | 3d |
| t_41477d65 | fix: political entity classification + merge eligibility in entity resolution | 4d |

### Track B — Infrastructure
| Task ID | Title | Est. |
|---|---|---|
| t_1539a1df | infra: .env.example + docker-compose — add Memgraph service vars | 1d |
| t_f5f3022a | infra: add Memgraph (memgraph-mage) to docker-compose | 1d |
| t_cc1d9e27 | infra: GitHub Actions CI — lint + pytest + tsc on all services | 2d |

### Track C — Service Scaffolding (starts mid-phase)
| Task ID | Title | Est. |
|---|---|---|
| t_adcd4584 | arch: create services/search/ from scrapers/ — REST API + job queue | 3d |
| t_4273d9cf | arch: expose graphgen as REST API (POST /build) | 2d |

### Phase 1 Gate Tasks (unblock after Track A done)
| Task ID | Title | Blocked By |
|---|---|---|
| t_b267bf42 | feat: full corpus ingest — EP + UK legislation into Neo4j+pgvector | all Track A |
| t_639d6b60 | eval: RAGAS re-evaluation of hybrid GraphRAG | t_b267bf42 |

---

## 🏗️ Phase 2 — Service Architecture (weeks 5–10)

**Goal:** decompose the monolith into clean services. Agent becomes the sole frontend entry point with per-session Memgraph overlay. Exit criterion: agent can answer multi-hop EP debate queries end-to-end via REST + WebSocket, with live graph viz.

### Track A — GraphRAG & Agent
| Task ID | Title | Blocked By |
|---|---|---|
| t_848f98cd | arch: expose graphrag as REST API (POST /retrieve) | t_66aa8b14, t_930b0b51 |
| t_6f42543e | arch: extract agent service (services/agent/) | t_848f98cd, t_4273d9cf |
| t_71bc0adb | arch: session graph manager — Memgraph CRUD + Neo4j seed | t_f5f3022a |
| t_88cdb519 | arch: agent WebSocket — live session graph delta stream | t_6f42543e, t_71bc0adb |
| t_9afb9e25 | arch: session write-back from Memgraph → Neo4j | t_6f42543e, t_71bc0adb |
| t_b9101d18 | feat: graphrag get_path tool (LKIF multi-hop traversal) | t_b2627d7f |

### Track B — Data Expansion (parallel)
| Task ID | Title | Blocked By |
|---|---|---|
| t_e842525d | feat: Bundestag DIP API scraper | t_adcd4584 |
| t_d3dc1591 | feat: UK Hansard API scraper | t_adcd4584 |
| t_25a47951 | feat: Bundestag DIP scraper in services/search/ | t_adcd4584, t_3e7c1079 |
| t_3e7c1079 | feat: multilingual GLiNER2 (German + French) | — |
| t_56b6a731 | feat: cross-parliament document reference following | t_adcd4584 |

---

## 🖥️ Phase 3 — Frontend v2 + Export (weeks 11–16)

**Goal:** production-quality frontend: split-pane chat+graph, live reasoning trace, investigation sessions, one-click export. Exit criterion: journalist/analyst can complete a research session and export a shareable graph.

| Task ID | Title | Blocked By |
|---|---|---|
| t_ab74b2cf | feat: frontend v2 — split-pane chat + D3.js graph canvas | t_6f42543e |
| t_948af8c4 | feat: reasoning trace sidebar — collapsible DAG of agent steps | t_ab74b2cf, t_88cdb519 |
| t_3094ed80 | feat: investigation sessions — named, saved, resumable | t_ab74b2cf |
| t_cc183272 | feat: graph export suite — PNG/SVG/HTML/permalink/cards | t_ab74b2cf |

---

## 🔒 Phase 4 — Hardening + Pilot (weeks 17–24)

**Goal:** production hardening for first pilot customers. Security, monitoring, documentation.

| Task ID | Title | Blocked By |
|---|---|---|
| t_62436ecb | infra: API auth + rate limiting for all services | t_6f42543e |
| t_e275bf27 | eval: formal benchmark — hybrid GraphRAG vs RAG vs LDA | t_b267bf42, t_6f42543e |
| t_78d9a39f | feat: monitoring — real-time ingest pipeline + diff alerting | t_6f42543e |
| t_ed277be9 | docs: developer documentation site (MkDocs Material) | — |

---

## Timeline Summary

```
Week  1- 4  Phase 1: Foundation Hardening
              [A] bug fixes in parallel
              [B] infra: Memgraph + CI/CD
              [C] service scaffolding begins

Week  5- 8  Phase 2a: Service Architecture
              agent service extracted
              Memgraph session layer wired
              graphrag + graphgen exposed as APIs

Week  9-12  Phase 2b: Data Expansion
              Bundestag + Hansard scrapers
              multilingual GLiNER2
              reference following

Week 13-16  Phase 3: Frontend v2
              split-pane UI
              reasoning trace
              export suite

Week 17-20  Phase 4: Hardening
              auth + rate limiting
              formal evaluation
              monitoring
              docs site

Week 21-24  Phase 5: Pilot
              first 5-10 pilot customers
              feedback loop
              Series A preparation
```

---

## Links

[[product-architecture-v2]] [[kanban-plan]] [[technical-debt]] [[business-plan]] [[gtm-strategy]]
