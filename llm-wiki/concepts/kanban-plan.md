---
title: Kanban Task Registry — Graphknows v2
created: 2026-05-03
updated: 2026-05-25
type: concept
tags: [planning, kanban, admin]
confidence: high
---

# Kanban Task Registry

---

## Phase — Universal Agent Memory (2026-05-25)

### Dead Code Removal + GDS Consolidation

| ID | Title | Status | Brief | Priority |
|---|---|---|---|---|
| DEAD-001 | Remove dead HTTP API and old MCP server | ✅ Done | `.briefs/DEAD-001-remove-dead-code.md` | HIGH |
| GDS-001 | Consolidate GDS paths; expose run_gds in MCP flush | ✅ Done | `.briefs/GDS-001-consolidate-gds-paths.md` | MEDIUM |
| GRAPHGEN-001 | Make graphgen domain-agnostic (prompt + taxonomy) | ✅ Done | `.briefs/GRAPHGEN-001-domain-agnostic.md` | HIGH |
| GRAPHGEN-002 | Expose corpus_ingest as MCP tool | ✅ Done | `.briefs/GRAPHGEN-002-corpus-ingest-mcp.md` | MEDIUM |

**Dependency order:** DEAD-001 → GDS-001 (must delete mcp/server.py before wiring). GRAPHGEN-001 and GRAPHGEN-002 are independent — run in parallel after DEAD-001 lands.

---

> Single source of truth for all Kanban tasks. Ordered by phase → dependency order within phase. IDs are real `hermes kanban` task IDs.

---

## Phase 1 — Foundation Hardening

### Track A: Bug Fixes

| ID | Title | Status | Priority | Est. | Blocks |
|---|---|---|---|---|---|
| t_66aa8b14 | fix: pgvector retrieval end-to-end | running | HIGH | 2d | t_b267bf42 |
| t_930b0b51 | fix: entity label collapse (CONCEPT=72%) | running | HIGH | 3d | t_b267bf42 |
| t_d5f42600 | fix: community_id null on Topic nodes | running | MEDIUM | 2d | t_b267bf42 |
| t_40558195 | feat: NeoStore.get_chunks_for_entity() | running | MEDIUM | 2d | t_639d6b60 |
| t_b2627d7f | fix: typed LKIF_REL edges via DSPy to Neo4j | todo | HIGH | 3d | t_b9101d18 |
| t_41477d65 | fix: political entity classification + merge eligibility | todo | HIGH | 4d | t_b267bf42 |

### Track B: Infrastructure

| ID | Title | Status | Priority | Est. | Blocks |
|---|---|---|---|---|---|
| t_1539a1df | infra: .env.example + Memgraph vars in docker-compose | todo | HIGH | 1d | t_f5f3022a |
| t_f5f3022a | infra: add Memgraph to docker-compose | ready | HIGH | 1d | t_71bc0adb |
| t_cc1d9e27 | infra: GitHub Actions CI (lint + pytest + tsc) | todo | MEDIUM | 2d | — |

### Track C: Service Scaffolding

| ID | Title | Status | Priority | Est. | Blocks |
|---|---|---|---|---|---|
| t_adcd4584 | arch: create services/search/ from scrapers/ | running | HIGH | 3d | t_25a47951, t_d3dc1591, t_56b6a731 |
| t_4273d9cf | arch: expose graphgen as REST API (POST /build) | running | HIGH | 2d | t_6f42543e |

### Phase 1 Gate Tasks (blocked)

| ID | Title | Status | Blocked By |
|---|---|---|---|
| t_b267bf42 | feat: full corpus ingest (EP + UK legislation) | todo | t_66aa8b14, t_930b0b51, t_d5f42600, t_41477d65 |
| t_639d6b60 | eval: RAGAS re-evaluation | todo | t_b267bf42, t_40558195 |

---

## Phase 2 — Service Architecture

### Track A: GraphRAG & Agent

| ID | Title | Status | Priority | Blocked By |
|---|---|---|---|---|
| t_848f98cd | arch: graphrag REST API (POST /retrieve) | todo | HIGH | t_66aa8b14, t_930b0b51 |
| t_6f42543e | arch: extract agent service (services/agent/) | todo | HIGH | t_848f98cd, t_4273d9cf |
| t_71bc0adb | arch: session graph manager — Memgraph CRUD + Neo4j seed | todo | HIGH | t_f5f3022a |
| t_88cdb519 | arch: agent WebSocket — live session graph delta stream | todo | HIGH | t_6f42543e, t_71bc0adb |
| t_9afb9e25 | arch: session write-back Memgraph → Neo4j | todo | MEDIUM | t_6f42543e, t_71bc0adb |
| t_b9101d18 | feat: graphrag get_path tool (LKIF multi-hop) | todo | MEDIUM | t_b2627d7f |

### Track B: Data Expansion

| ID | Title | Status | Priority | Blocked By |
|---|---|---|---|---|
| t_3e7c1079 | feat: multilingual GLiNER2 (German + French) | todo | HIGH | — |
| t_e842525d | feat: Bundestag DIP API scraper (raw) | running | MEDIUM | — |
| t_25a47951 | feat: Bundestag DIP scraper in services/search/ | todo | MEDIUM | t_adcd4584, t_3e7c1079 |
| t_d3dc1591 | feat: UK Hansard API scraper | todo | MEDIUM | t_adcd4584 |
| t_56b6a731 | feat: cross-parliament document reference following | todo | LOW | t_adcd4584 |

---

## Phase 3 — Frontend v2

| ID | Title | Status | Priority | Blocked By |
|---|---|---|---|---|
| t_ab74b2cf | feat: frontend v2 — split-pane chat + D3.js graph | todo | HIGH | t_6f42543e |
| t_948af8c4 | feat: reasoning trace sidebar (live DAG of agent steps) | todo | HIGH | t_ab74b2cf, t_88cdb519 |
| t_3094ed80 | feat: investigation sessions (named, saved, resumable) | todo | HIGH | t_ab74b2cf |
| t_cc183272 | feat: graph export suite (PNG/SVG/HTML/permalink) | todo | HIGH | t_ab74b2cf |

---

## Phase 4 — Hardening + Pilot

| ID | Title | Status | Priority | Blocked By |
|---|---|---|---|---|
| t_62436ecb | infra: API auth + rate limiting | todo | HIGH | t_6f42543e |
| t_e275bf27 | eval: formal benchmark (GraphRAG vs RAG vs LDA) | todo | HIGH | t_b267bf42, t_6f42543e |
| t_78d9a39f | feat: monitoring + diff alerting | todo | MEDIUM | t_6f42543e |
| t_ed277be9 | docs: developer documentation site | todo | LOW | — |

---

## DONE

| What | When |
|---|---|
| Thesis full corpus run (6,377 nodes, 20,661 edges) | Phase 0 |
| Node2Vec modularity improvement (0.748 → 0.823) | Phase 0 |
| Leiden hierarchical community detection | Phase 0 |
| LLM community summarization pipeline | Phase 0 |
| IterativeOrchestrator refactor (CC 63 → ≤10) | Phase 0 |
| LDA baseline comparison | Phase 0 |
| Neo4j adapter | Phase 0 |
| pgvector schema setup | Phase 0 |
| PR#1: deleted orphaned LlamaIndex src/, label fallback→UNKNOWN, IterativeOrchestrator decomposed | Phase 1 |
| Product architecture v2 document | Phase 1 planning |
| Memgraph session graph layer decision | Phase 1 planning |

---

## Links

[[project-roadmap]] [[technical-debt]] [[product-architecture-v2]] [[administration]]
