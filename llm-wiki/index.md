# Wiki Index

> Political Intelligence & Knowledge Graph Research — Graphknows v2 knowledge base.
> Read this first to find relevant pages for any query.
> Last updated: 2026-05-04 | Total pages: 20

---

## ⚡ Start Here for Development

- [[product-architecture-v2]] — v2 architecture overview: service decomposition, Memgraph session graphs, frontend, rationale
- [[project-roadmap]] — **4-phase roadmap with exit criteria, timeline, and all task IDs**
- [[kanban-plan]] — **full Kanban registry** — all tasks with IDs, status, dependencies

---

## Entities

- [[graphgen-project]] — The GraphGen Python package: KG generation from political text, tech stack, key results, infrastructure

---

## Concepts — Architecture

- [[graphgen-architecture]] — Full pipeline architecture: DOC→SEGMENT→CHUNK→ENTITY→TOPIC, PipelineContext bus, module tree, config system
- [[pipeline-stages]] — Each pipeline stage in detail: OntologyLabelExtractor, chunking, GLiNER2, DSPy, graph assembly, Node2Vec, Leiden, summarization
- [[knowledge-graph-construction]] — KG construction methodology, ontology schema, scale-free topology, entity resolution, canonicalization
- [[graphrag-and-downstream]] — GraphKnows hybrid GraphRAG agent, Neo4j+pgvector dual retrieval, pending tasks, RAGAS plan

- [[architecture-reference]] — **canonical technical reference**: ASCII service map, full data-flow, API contracts, env vars, local startup commands, tunnel setup
- [[product-roadmap-v2]] — **strategic roadmap v2**: shipped PRs, stability gates, prioritised next features, horizon bets, GTM beachhead

### Service Design (v2)

- [[service-design-agent]] — `services/agent/` spec: HTTP API, tools (6), session lifecycle, Memgraph CRUD, write-back
- [[service-design-search]] — `services/search/` spec: source adapters, job queue, idempotent ingestion, reference resolver
- [[frontend-v2-design]] — `apps/web/` v2: split-pane layout, D3.js graph, reasoning trace, export suite, investigation sessions

---

## Concepts — Product & Business

- [[business-plan]] — Problem, solution, target customers, value prop, revenue model, competitive landscape, moat, funding
- [[market-analysis]] — TAM/SAM/SOM, comparable SaaS pricing, EU regulatory demand, buyer personas
- [[project-roadmap]] — Phase 0–4 roadmap: Foundation → Architecture → Frontend → Hardening → Pilot

---

## Concepts — Operations

- [[technical-debt]] — 10-item debt register with severity, description, and remediation (label collapse, community_id, pgvector, etc.)
- [[administration]] — Repo structure, branch/PR workflow, config management, key commands, infrastructure endpoints (Neo4j 7687, Memgraph 7688, pgvector 5432)
- [[kanban-plan]] — Full task registry: all phases, IDs, status, dependencies

---

## Queries

- [[gtm-strategy]] — Three-phase GTM: EU Parliament → national parliaments → regulatory bodies

---

## Raw Sources

- raw/papers/graphgen-thesis-synthesis.md — Immutable synthesis of thesis findings and corpus results
- [[thesis-findings-and-learnings]] — Empirical results, what worked/didn't, open questions, academic framing
- [[test-results-frontend]] — Frontend v2 smoke test results (8/8 pass after fixes)
- [[eu-source-taxonomy]] — Full taxonomy of EU public record types: 12 institutions, 25+ source types, PIV/difficulty ratings, adapter priority order
- [[graph-visualisation-diagnosis]] — Root cause analysis: missing edges, no topic nodes, empty citations, node label issues, RAGAS plan
