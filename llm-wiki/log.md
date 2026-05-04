# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: ## [YYYY-MM-DD] action | subject
> Rotate when >500 entries.

## [2026-05-03] create | Wiki initialized
- Domain: Political Intelligence & Knowledge Graph Research (GraphGen project)
- Structure: SCHEMA.md, index.md, log.md, raw/, entities/, concepts/, comparisons/, queries/

## [2026-05-03] create | entities/graphgen-project.md
- GraphGen project entity page: tech stack, key results, infrastructure

## [2026-05-03] create | concepts/graphgen-architecture.md
- Full pipeline architecture: DOC->SEGMENT->CHUNK->ENTITY->TOPIC, PipelineContext, modules

## [2026-05-03] create | concepts/pipeline-stages.md
- All 10 pipeline stages with rationale and parameters

## [2026-05-03] create | concepts/knowledge-graph-construction.md
- KG construction: ontology, scale-free topology, entity resolution, canonicalization

## [2026-05-03] ingest | raw/papers/graphgen-thesis-synthesis.md
- Immutable synthesis of thesis corpus, results, evaluation findings

## [2026-05-03] create | concepts/thesis-findings-and-learnings.md
- Core claim, empirical results table, what worked/didn't, open questions, academic framing

## [2026-05-03] create | concepts/graphrag-and-downstream.md
- GraphKnows GraphRAG agent, Neo4j+pgvector dual retrieval, pending task tracker

## [2026-05-03] create | concepts/business-plan.md
- Full business plan: problem, solution, customers, revenue model, competitive landscape

## [2026-05-03] create | concepts/market-analysis.md
- TAM/SAM/SOM, comparable SaaS pricing, EU regulatory demand, buyer personas

## [2026-05-03] create | queries/gtm-strategy.md
- Three-phase GTM strategy: EU Parliament -> national parliaments -> regulatory bodies

## [2026-05-03] create | concepts/project-roadmap.md
- Phase 0-3 roadmap with task IDs, dependencies, effort estimates

## [2026-05-03] create | concepts/technical-debt.md
- 10-item debt register: label collapse, community_id, pgvector, entity resolution, etc.

## [2026-05-03] create | concepts/administration.md
- Repo structure, PR workflow, config management, key commands, infrastructure

## [2026-05-03] create | concepts/kanban-plan.md
- Full Kanban board with in-progress, blocked, backlog, done columns

## [2026-05-03] update | index.md
- Consolidated canonical index after parallel wiki population

## [2026-05-04] create | concepts/product-architecture-v2.md
- Full product architecture redesign: search service (from scrapers), agent service (from graphrag), service decomposition, session-scoped context graph, frontend viz exports, market/academic validation synthesis


## [2026-05-04] update | concepts/product-architecture-v2.md
- Added Memgraph as session graph layer (replaces Redis/in-memory Neo4j); full technical rationale, Docker config, Cypher schema, MAGE algorithms

## [2026-05-04] update | concepts/graphrag-and-downstream.md
- Added Memgraph session graph section: architecture, lifecycle, Cypher differences

## [2026-05-04] create | kanban: t_f5f3022a infra-memgraph-docker — Memgraph in docker-compose for agent service
