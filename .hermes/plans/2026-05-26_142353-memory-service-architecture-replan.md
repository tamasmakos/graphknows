# Graphknows memory service architecture replan

Date: 2026-05-26
Repo: `/root/graphknows`
Author: Hermes PM

## Goal

Re-plan the graphknows package around the user's updated architectural direction:
- repo structure must represent the memory service
- organize around `memory/` sublayers: STM, LTM, SEMANTIC
- move shared concepts like taxonomy into domain-level modules
- route batch corpus ingestion through STM first
- run all feature engineering during promotion to LTM using Neo4j GDS
- remove legacy/compatibility files rather than preserving backward compatibility
- maximize modularity with proper builders, dependency injection, and clean config ownership

## Current diagnosis

The repo currently contains **three overlapping architectures**:

1. **Legacy corpus / HybridRAG architecture**
   - centered on `graphknows/container.py`, `graphknows/__init__.py`, `graphknows/retrieval/retriever.py`
   - still references legacy pgvector tables (`chunks`, `entities`) and broad tool surfaces

2. **STM/LTM memory architecture**
   - centered on `graphknows/memory/*`, `graphknows/mcp/stdio_server.py`, `graphknows/cli/memory.py`
   - this is closest to the intended service

3. **Vendored graphgen pipeline architecture**
   - centered on `graphknows/graphgen/*`
   - still acts like a parallel ingestion system instead of an implementation detail behind the memory service

The package map reflects implementation history more than service boundaries.

## Key architectural problems

### 1. Service shape is not represented by the package structure
- `memory/` is not the clear architectural center
- `retrieval/` holds core memory behavior that should conceptually belong under memory
- `graphgen/` remains a first-class public architecture
- `models/` contains stale pipeline/session/frontend-era models

### 2. Batch corpus ingestion bypasses STM
Observed public entrypoints:
- MCP `corpus_ingest` delegates directly to `graphknows.graphgen.ingest.run_ingestion`
- `GraphKnowsContainer` ingestion delegates directly to graphgen

This conflicts with the target architecture where all ingestion should enter STM first.

### 3. Feature engineering / enrichment is fragmented
Current enrichment logic is split across:
- `memory/flush.py`
- `retrieval/stores/neo4j.py:summarize_topics()`
- `retrieval/stores/neo4j.py:run_gds_enrichment()`
- `feature_engineering.py`
- graphgen pipeline community/summarization logic

There is no single reusable promotion/enrichment pipeline.

### 4. MCP and CLI are too thick
`mcp/stdio_server.py` and `cli/memory.py` both:
- load envs
- create stores directly
- duplicate health/doctor logic
- own business wiring instead of calling shared services

### 5. Builders / DI are missing
Object construction is scattered across:
- `container.py`
- `mcp/stdio_server.py`
- `cli/memory.py`
- `memory/adapter.py`

This causes duplicated lifecycle logic and poor modularity.

### 6. Schema ownership and labels are inconsistent
There are competing schema/query conventions across:
- `schema.py`
- `retrieval/stores/neo4j.py`
- `retrieval/stores/memgraph.py`
- recall/forget/MCP/CLI queries

Labels are mixed (`ENTITY` vs `Entity`, `TOPIC` vs `Topic`, etc.).

### 7. Shared domain concepts live in the wrong namespace
- `memory/extractor.py` imports taxonomy from `graphknows.graphgen.config.taxonomy`
- shared taxonomy/domain concepts should live under `graphknows/domain/`

### 8. Retrieval/stores contain business logic
`retrieval/stores/neo4j.py` is not just a store adapter. It also handles:
- relation extraction orchestration
- topic summarization
- GDS enrichment
- topic writes
- provenance/segment reads

That is too much responsibility for an adapter.

### 9. The async bridge `_run` leaks across layers
`memory/ingestor.py` owns a hidden background loop and `_run()`, but STM/LTM recall and other modules import it across boundaries.

### 10. Legacy surfaces still dominate the public API
Examples:
- `mcp_ltm.py`
- broad `GraphKnowsContainer` tool surface
- stale legacy pgvector methods
- `forget.py`
- stale `models/` package contents

## Proposed target architecture

```text
graphknows/
  domain/
    taxonomy.py
    ids.py
    labels.py
    types.py

  config/
    settings.py
    env.py
    doctor.py

  builders/
    adapters.py
    services.py
    app.py

  adapters/
    graph/
      memgraph_stm.py
      neo4j_ltm.py
    vector/
      pgvector.py
    extraction/
      gliner2.py
      relations.py
    llm/
      openrouter.py

  memory/
    models/
      evidence.py
      requests.py
      results.py
      promotion.py

    stm/
      ingest.py
      dialog.py
      recall.py
      service.py

    ltm/
      promote.py
      enrich.py
      recall.py
      forget.py
      service.py

    semantic/
      embed.py
      fusion.py
      topics.py
      service.py

    workflows/
      corpus_ingest.py

  mcp/
    server.py
    tools/
      stm.py
      ltm.py
      semantic.py
      admin.py
      corpus.py

  cli/
    memory.py
```

## Architectural rules

1. `memory/` is the service center.
2. All ingestion enters STM first.
3. All promotion to LTM goes through a single promotion pipeline.
4. All enrichment (community detection, centrality, topic summarization) happens during promotion using Neo4j GDS.
5. `graphgen/` is not a parallel public architecture anymore; surviving pieces become adapters or one-off migration utilities.
6. `mcp/` and `cli/` are thin transports only.
7. No import-time `load_dotenv()`.
8. No service/business module reads `os.environ` directly.
9. Adapters do infra only; service logic stays in STM/LTM/SEMANTIC services.
10. Legacy compatibility shims are removable.

## Recommended phased sequence

### Phase 1 — define target boundaries
- decide canonical package map
- decide canonical label/relationship naming
- define allowed import directions

### Phase 2 — unify config/env/builders
- create single runtime settings root
- create env resolution utility
- create shared doctor/health utility
- create adapter/service builders

### Phase 3 — move shared domain concepts
- create `graphknows/domain/taxonomy.py`
- add shared IDs/labels/types
- remove memory -> graphgen taxonomy dependency

### Phase 4 — replace stale models with memory/domain models
- create `memory/models/*`
- retire stale `models/pipeline.py` and `models/session.py` surfaces if unused

### Phase 5 — split adapters from business services
- carve low-level DB logic into adapters
- remove business orchestration from `retrieval/stores/*`

### Phase 6 — rebuild STM layer
- move STM ingest/dialog/recall into `memory/stm/*`
- remove cross-layer dependence on `_run`

### Phase 7 — rebuild LTM promotion and enrichment
- create one promotion pipeline
- create one reusable GDS-based enrichment pipeline
- remove parallel enrichment paths

### Phase 8 — route batch corpus ingestion through STM first
- implement `memory/workflows/corpus_ingest.py`
- batch docs -> STM -> promote -> enrich

### Phase 9 — thin transports
- refactor MCP to tool modules over services/builders
- refactor CLI to reuse the same services/builders

### Phase 10 — delete legacy surfaces and enforce boundaries
- remove compatibility shims and stale APIs
- add import-boundary checks in CI

## Files likely to change

High probability:
- `graphknows/container.py`
- `graphknows/__init__.py`
- `graphknows/mcp/stdio_server.py`
- `graphknows/cli/memory.py`
- `graphknows/config.py`
- `graphknows/schema.py`
- `graphknows/feature_engineering.py`
- `graphknows/forget.py`
- `graphknows/recall.py`
- `graphknows/memory/*`
- `graphknows/retrieval/*`
- `graphknows/graphgen/*`
- `graphknows/models/*`
- `graphknows/migrations/*`

Likely new files/directories:
- `graphknows/domain/*`
- `graphknows/config/*`
- `graphknows/builders/*`
- `graphknows/adapters/*`
- `graphknows/memory/models/*`
- `graphknows/memory/stm/*`
- `graphknows/memory/ltm/*`
- `graphknows/memory/semantic/*`
- `graphknows/memory/workflows/*`
- `graphknows/mcp/tools/*`

## Validation / acceptance approach

Before and during implementation:
- import caller audit for legacy modules
- label/schema consistency audit over all Cypher code
- focused unit tests by target package
- live validation of STM -> LTM promotion
- live validation that batch corpus ingestion actually enters STM first
- live validation that a single enrichment pipeline is used by both session flush and corpus ingestion

Core acceptance criteria for the architecture program:
1. Memory package is clearly layered into STM/LTM/SEMANTIC.
2. Batch ingestion enters STM first.
3. One enrichment path exists and is reused.
4. MCP and CLI are thin wrappers over shared services.
5. Config and store construction are centralized.
6. No active codepath depends on legacy `chunks/entities` tables or compatibility shims.
7. Import-boundary rules are enforced in CI.

## Risks

- Large refactor with possible need for graph data reset or migration
- Neo4j GDS becomes a hard dependency for enrichment
- Some graphgen code is useful, but must be carefully extracted instead of blindly deleted
- Public API breakage is intentional and should be handled as deliberate cleanup

## Open questions

1. Should graphgen survive only as an internal adapter source, or should it be removed entirely after extraction of useful components?
2. Do we want `GraphKnowsContainer` to survive as a narrow app builder, or remove it entirely in favor of a new memory app builder?
3. Should stale `models/` be deleted wholesale or pruned incrementally after usage audit?
4. Is a hard data reset acceptable while normalizing labels/schema, or must we script a migration?
