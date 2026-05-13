# Wiki Index — Graphknows

> Deep research agent for political intelligence.
> Last updated: 2026-05-12 | Pages: 26

---

## architecture/ — Authoritative Design Docs

| Page | Summary |
|---|---|
| [[architecture/product-architecture]] | Vision, service map, agent loop, infrastructure roles, non-goals |
| [[architecture/agent-design]] | Deep research agent: generate→search→reflect loop, novelty stopping, InvestigativeState, WS events |
| [[architecture/dual-write-retrieval]] | DualWriter (Memgraph + pgvector), RRF hybrid retrieval, HybridRetriever interface |
| [[architecture/reference]] | Full service topology, API contracts, ports, env vars, local dev setup |
| [[architecture/service-agent]] | services/agent/ internal design: LangGraph nodes, session graph, HTTP clients |
| [[architecture/service-search]] | services/search/ design: BaseSource, adapter pattern, EU source adapters |

---

## adr/ — Architecture Decision Records

| # | Page | Decision | Status |
|---|---|---|---|
| 001 | [[adr/001-langgraph-agent]] | Replace ReAct loop with LangGraph StateGraph | accepted |
| 002 | [[adr/002-investigative-workflow]] | 3-phase investigative workflow (Scope→Investigate→Communicate) | accepted |
| 003 | [[adr/003-services-refactor]] | Decompose monolith into search/graphgen/graphrag/agent microservices | accepted |
| 004 | [[adr/004-ontology-patch]] | Ontology injection bug + 3-phase patch runner | accepted |
| 005 | [[adr/005-pluggable-session-graph-backend]] | Pluggable session KG backend (Memgraph/Neo4j+GDS) + NodeFeatures for reasoning | accepted |

---

## product/ — Vision, Roadmap, Business

| Page | Summary |
|---|---|
| [[product/roadmap]] | Current sprint, next features, horizon — living document |
| [[product/business-plan]] | Business model, target users, monetisation |
| [[product/market-analysis]] | Competitive landscape, market sizing |
| [[product/frontend-design]] | Frontend v2 spec: split-pane chat + D3 graph canvas + export suite |
| [[product/gtm-strategy]] | Go-to-market: beachhead users, channels, pricing |

---

## research/ — Domain Knowledge & Evaluation

| Page | Summary |
|---|---|
| [[research/eu-source-taxonomy]] | Full taxonomy of EU public record types, endpoints, formats, political value |
| [[research/kg-construction]] | KG construction methodology: GLiNER2, DSPy, Leiden, ontology |
| [[research/thesis-findings]] | ELTE MSc thesis findings: EP debate corpus, Node2Vec, modularity results |
| [[research/ragas-evaluation]] | RAGAS eval results (baseline): context_precision, faithfulness, relevancy |

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
| [[entities/graphgen-project]] | GraphGen pipeline: history, capabilities, current role |

---

## raw/ — Immutable Source Material

| Page | Summary |
|---|---|
| [[raw/papers/graphgen-thesis-synthesis]] | Synthesised GraphGen MSc thesis — source material |
