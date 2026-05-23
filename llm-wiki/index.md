# Wiki Index — Graphknows

> Deep research agent for political intelligence.
> Last updated: 2026-05-23 | Pages: 28

---

## architecture/ — Authoritative Design Docs

| Page | Summary |
|---|---|
| [[architecture/product-architecture]] | Vision, service map, agent loop, infrastructure roles, non-goals |
| [[architecture/agent-design]] | Deep research agent: generate→search→reflect loop, novelty stopping, InvestigativeState, WS events |
| [[architecture/dual-write-retrieval]] | DualWriter (Memgraph + pgvector), RRF hybrid retrieval, HybridRetriever interface |
| [[architecture/reference]] | Full service topology, API contracts, ports, env vars, local dev setup |
| [[architecture/service-agent]] | services/agent/ internal design: LangGraph nodes, session graph, HTTP clients |
| [[architecture/service-search]] | services/search/ design: BaseSource, connector pattern, EU source connectors |
| [[architecture/generic-memory-engine-discovery]] | Deep prototype findings for generic memory objects, raw-vs-distilled lanes, and cross-session promotion |
| [[architecture/generic-agent-adapter-discovery]] | Repo-level findings on a host-agnostic adapter contract, generic event surfaces, and recall-lane gaps |
| [[architecture/generic-agent-adapter-implementation-plan]] | Repo-level implementation wave, dependency graph, and acceptance gate for the generic adapter |

---

## adr/ — Architecture Decision Records

| # | Page | Decision | Status |
|---|---|---|---|
| 001 | [[adr/001-langgraph-agent]] | Replace ReAct loop with LangGraph StateGraph | accepted |
| 002 | [[adr/002-investigative-workflow]] | 3-phase investigative workflow (Scope→Investigate→Communicate) | accepted |
| 003 | [[adr/003-services-refactor]] | Decompose monolith into search/graphknows/agent microservices | accepted |
| 004 | [[adr/004-ontology-patch]] | Ontology injection bug + 3-phase patch runner (targets Neo4j legacy corpus) | accepted |
| 005 | [[adr/005-pluggable-session-graph-backend]] | Pluggable session KG backend (Memgraph/Neo4j+GDS) + NodeFeatures for reasoning | accepted |
| 006 | [[adr/006-ontology-guided-extraction]] | Domain-agnostic ontology-guided extraction: GLiNER2 labels from ontology + DSPy with schema context | accepted |

---

## product/ — Vision, Roadmap, UX

| Page | Summary |
|---|---|
| [[product/roadmap]] | Current sprint, next features, horizon — living document |
| [[product/frontend-design]] | Frontend spec: split-pane chat + D3 graph canvas + export suite |

> Speculative future planning docs (business-plan, market-analysis, gtm-strategy) quarantined to `product/future/` — not grounded in current code state.

---

## research/ — Domain Knowledge & Evaluation

| Page | Summary |
|---|---|
| [[research/eu-source-taxonomy]] | Full taxonomy of EU public record types, endpoints, formats, political value |
| [[research/kg-construction]] | KG construction methodology: GLiNER2, DSPy, Leiden, ontology |
| [[research/thesis-findings]] | ELTE MSc thesis findings: EP debate corpus, Node2Vec, modularity results |

> RAGAS evaluation results removed — pre-pivot baseline (2025-05-04, old shared-KG architecture), not comparable to current per-session design.

---

## ops/ — Operations & Infra

| Page | Summary |
|---|---|
| [[ops/administration]] | Service startup, tunnel, environment setup |
| [[ops/technical-debt]] | Known tech debt items, prioritised |

---

## entities/ — Tools, Models, Services

| Page | Summary |
|---|---|
| [[entities/graphknows-service]] | services/graphknows consolidated package: graphgen pipeline, retrieval, ontology, MCP |

---

## raw/ — Immutable Source Material

| Page | Summary |
|---|---|
| [[raw/papers/graphgen-thesis-synthesis]] | Synthesised GraphGen MSc thesis — source material |
