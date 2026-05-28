# graphknows Architecture Cleanup Plan

**Date**: 2026-05-26  
**Author**: Hermes PM  
**Status**: DRAFT  

---

## Executive Summary

The graphknows repo currently exhibits **three overlapping architectures**:

1. **Legacy HybridRAG container architecture** — corpus retrieval patterns, batch-oriented
2. **New STM/LTM memory service architecture** — real-time agent memory paths
3. **Vendored graphgen batch-pipeline architecture** — graph generation from documents

This creates:
- Confused boundaries (what's ingestion? what's enrichment? what's retrieval?)
- Leaky abstractions (business logic in stores, taxonomy in graphgen config)
- Duplicated responsibilities (MCP vs CLI, schema vs store initialization)
- Misplaced modules (feature_engineering, recall, forget at root level)

**Core principle for cleanup**: The repo structure must mirror the memory service.

---

## Target Service Architecture

```
graphknows/
├── memory/
│   ├── stm/          # Short-term memory service layer
│   │   ├── ingest.py
│   │   ├── recall.py
│   │   └── schema.py
│   ├── ltm/          # Long-term memory service layer
│   │   ├── promotion.py   # STM→LTM flush
│   │   ├── enrichment.py  # GDS, Node2Vec, PageRank
│   │   ├── recall.py
│   │   └── schema.py
│   └── semantic/     # Semantic memory layer (topics, embeddings)
│       ├── topics.py
│       └── embeddings.py
│
├── domain/           # Shared domain concepts
│   ├── taxonomy.py
│   ├── entities.py
│   └── schema.py     # Canonical graph label definitions
│
├── infrastructure/   # Adapters and stores
│   ├── neo4j.py
│   ├── memgraph.py
│   ├── pgvector.py
│   └── embedder.py
│
├── ingestion/        # Document parsing only
│   └── parsers/
│
├── transports/       # Thin protocol layers
│   ├── mcp/
│   │   ├── server.py        # Bootstrap only
│   │   └── tools/
│   │       ├── stm_tools.py
│   │       ├── ltm_tools.py
│   │       └── diagnostics_tools.py
│   └── cli/
│       ├── main.py
│       └── commands/
│
├── graphgen/         # Batch corpus pipeline (legacy)
│   └── ...           # Keep isolated, clearly marked as batch-only
│
├── config.py         # Runtime configuration builder
├── builder.py        # Service composition root
└── __init__.py       # Public library API
```

---

## Structural Problems Inventory

### 1. **Root-level modules that don't belong**

| File | Why it's wrong | Target location |
|------|----------------|-----------------|
| `feature_engineering.py` | Neo4j LTM enrichment, not generic | `memory/ltm/enrichment.py` |
| `recall.py` | Entity-based LTM recall | `memory/ltm/recall.py` |
| `forget.py` | LTM pruning/cleanup operations | `memory/ltm/forget.py` |
| `schema.py` | Graph DDL, not config | `domain/schema.py` |
| `mcp_ltm.py` | Backwards-compat shim | Remove or keep as explicit shim |

### 2. **MCP server monolith**

`mcp/stdio_server.py` — 797 lines containing:
- STM tools
- LTM tools
- Corpus ingest bridge
- Connection helpers
- Diagnostics
- Formatting utilities

**Problem**: No separation of concerns, hard to test tools in isolation.

### 3. **Taxonomy boundary leak**

- `graphgen/config/taxonomy.py` contains shared domain logic
- `memory/extractor.py` imports from graphgen config namespace
- MCP `corpus_ingest` tries to pass `taxonomy_file` but it's likely ignored

**Problem**: Shared domain concept hidden in pipeline-specific config.

### 4. **Stores contain business logic**

- `retrieval/stores/neo4j.py` has 1000+ lines
- Contains: connection pooling + schema init + GDS enrichment + recall queries + admin helpers
- `retrieval/stores/memgraph.py` similarly mixed

**Problem**: Adapters should be thin infrastructure wrappers, not service layers.

### 5. **Inconsistent graph label casing**

| Node Label | Appears as | Location |
|------------|-----------|----------|
| Entity | `:Entity` / `:ENTITY` | Mixed across codebase |
| Topic | `:Topic` / `:TOPIC` | Mixed across codebase |
| Session | `:SESSION` | Consistent (uppercase) |
| File | `:FILE` | Consistent (uppercase) |

**Problem**: `Entity` vs `ENTITY` in Cypher queries returns different results (0 rows silently).

### 6. **Duplicated health/diagnostics**

- `mcp/stdio_server.py` implements `memory_doctor`
- CLI has its own health checks
- Container initialization does implicit health checks
- No shared diagnostics service

### 7. **Config/env loading is scattered**

- `config.py` defines settings
- `mcp/stdio_server.py` loads `.env` at import time
- CLI commands load env separately
- Container doesn't use a builder pattern

**Problem**: No single composition root, side effects at import.

### 8. **Unclear ingestion paths**

Two ingestion paths exist:

**Path A**: Real-time session memory
- Text → `memory/ingestor.py` → STM (Memgraph + pgvector `stm_*`)
- Flush → `memory/flush.py` → LTM (Neo4j + pgvector `ltm_*`)

**Path B**: Batch corpus
- Documents → `graphgen/ingest.py` → Neo4j LTM directly (bypasses STM)

**Problem**: The two paths are disconnected. Batch corpus should flow through STM first for consistency.

### 9. **Legacy container surface**

`container.py` (596 lines) still exposes:
- Old segment/provenance helpers
- Cypher query tools
- Methods tied to old pgvector table schemas

**Problem**: Library surface includes stale APIs that don't align with new memory service.

### 10. **Models are stale**

`models/` contains:
- `pipeline.py` — graphgen pipeline models
- `session.py` — old session shapes
- `entity.py` — entity models
- `tools.py` — GraphKnowsTool wrapper

**Problem**: Models reflect old pipeline/frontend shapes, not current memory service boundaries.

### 11. **Migrations are orphaned**

- `migrations/002_stm_ltm_split.py` exists
- No migration runner
- No documented invocation path
- Unclear if it's admin script or supported feature

**Problem**: Half-implemented migration subsystem.

### 12. **Graphgen integration is shallow**

- `graphgen/` is a 9-subpackage tree
- Wrapped by `graphgen/ingest.py`
- MCP exposes `corpus_ingest` wrapper
- But treated as **bolted-on subsystem**, not first-class service path

**Problem**: Architectural asymmetry — session memory feels native, corpus feels imported.

---

## Critical Semantic Issues

### Issue A: Taxonomy override is broken

In `mcp/stdio_server.py`:
```python
config["taxonomy_file"] = taxonomy_file  # Top-level key
```

But `graphgen.config.settings.PipelineSettings` expects:
```python
settings.extraction.taxonomy_file  # Nested under extraction
```

**Result**: Taxonomy override is silently ignored in corpus ingestion.

### Issue B: GDS enrichment session_id is unused

- `feature_engineering.py` exports `run_feature_engineering(driver)` (no session_id)
- `memory/flush.py` calls `run_gds_enrichment(session_id)` (with session_id)
- But implementation appears to enrich **all LTM entities**, not session-scoped

**Result**: Semantic mismatch between API and implementation.

### Issue C: Label casing breaks queries

- Schema uses `:ENTITY` (uppercase)
- Some queries use `:Entity` (pascal)
- Cypher is case-sensitive
- Wrong casing → 0 rows, no error

**Result**: Silent query failures in production.

---

## Refactoring Principles

1. **Repo structure mirrors service structure**
   - `memory/stm`, `memory/ltm`, `memory/semantic` are first-class
   - Infrastructure adapters are thin
   - Transports (MCP, CLI) are thin protocol layers

2. **Shared domain concepts live in `domain/`**
   - Taxonomy
   - Graph labels (canonical casing)
   - Entity models

3. **Business logic lives in service layers, not stores**
   - Stores = connection + basic CRUD
   - Services = orchestration + business rules

4. **Single composition root**
   - `builder.py` constructs all services
   - Config loads once, explicitly
   - No import-time side effects

5. **Two ingestion paths become one**
   - Batch corpus should flow through STM first
   - LTM promotion becomes a **reusable pipeline** used by both paths

6. **Transports reuse services**
   - MCP tools call service methods
   - CLI commands call the same service methods
   - No business logic in transport layers

7. **Legacy compatibility is explicit**
   - Shims are clearly marked
   - Deprecated surfaces documented
   - Removal timeline specified

---

## Phased Execution Plan

### Phase 0: Normalize foundations (no breakage)

**Goal**: Establish canonical truth for shared concerns.

- [ ] Create `domain/schema.py` with canonical label casing rules
- [ ] Audit all Cypher queries for label casing violations
- [ ] Create `domain/taxonomy.py` and move taxonomy helpers
- [ ] Fix taxonomy override wiring in corpus_ingest
- [ ] Document the two ingestion paths explicitly

**Validation**: All existing tests pass, taxonomy override works in corpus_ingest.

---

### Phase 1: Extract services from stores

**Goal**: Separate infrastructure adapters from business logic.

- [ ] Create `infrastructure/` package
- [ ] Move thin connection/CRUD code to `infrastructure/{neo4j,memgraph,pgvector}.py`
- [ ] Create `memory/stm/service.py` with STM operations
- [ ] Create `memory/ltm/service.py` with LTM operations
- [ ] Create `memory/ltm/enrichment.py` and move `feature_engineering.py` logic
- [ ] Create `memory/ltm/recall.py` and move `recall.py` logic
- [ ] Update `retrieval/stores/*` to use new infrastructure adapters

**Validation**: Container still works, all tests pass.

---

### Phase 2: Rebuild composition root

**Goal**: Single builder pattern for all services.

- [ ] Create `builder.py` with service construction logic
- [ ] Refactor `container.py` to use builder
- [ ] Remove import-time `.env` loading from MCP server
- [ ] Create shared config loader
- [ ] Create shared diagnostics service

**Validation**: Container + MCP + CLI all use the same builder.

---

### Phase 3: Decompose MCP monolith

**Goal**: Thin, testable MCP tool modules.

- [ ] Create `transports/mcp/tools/stm_tools.py`
- [ ] Create `transports/mcp/tools/ltm_tools.py`
- [ ] Create `transports/mcp/tools/corpus_tools.py`
- [ ] Create `transports/mcp/tools/diagnostics_tools.py`
- [ ] Refactor `mcp/stdio_server.py` to import and register tools
- [ ] Each tool module depends on service layer, not stores

**Validation**: MCP server still works, tools are unit-testable.

---

### Phase 4: Unify ingestion paths

**Goal**: Batch corpus flows through STM first.

- [ ] Create `memory/ltm/promotion.py` with reusable STM→LTM pipeline
- [ ] Refactor `memory/flush.py` to use promotion pipeline
- [ ] Refactor `graphgen/ingest.py` to:
  1. Write to STM first
  2. Call promotion pipeline
  3. Run enrichment
- [ ] Document the unified ingestion flow

**Validation**: Both session flush and corpus ingest use the same LTM promotion logic.

---

### Phase 5: Refactor CLI to reuse services

**Goal**: CLI commands become thin wrappers over service methods.

- [ ] Refactor `cli/memory.py` to use builder + service layers
- [ ] Remove duplicated diagnostics/health check code
- [ ] Remove duplicated recall/forget logic

**Validation**: CLI commands work identically, no duplicated business logic.

---

### Phase 6: Remove legacy surface and enforce boundaries

**Goal**: Clean public API, enforced architectural rules.

- [ ] Remove or mark deprecated container methods
- [ ] Classify `forget.py` — move to `memory/ltm/forget.py` or remove
- [ ] Classify `migrations/` — real migration system or admin scripts?
- [ ] Remove `mcp_ltm.py` shim (or document explicitly as compat layer)
- [ ] Update `models/` to reflect service boundaries
- [ ] Add `tach.toml` boundary enforcement rules

**Validation**: Public API is clean, import boundaries enforced in CI.

---

## Success Criteria

After all phases:

1. **Repo structure mirrors service architecture**
   - `memory/stm`, `memory/ltm`, `memory/semantic` exist
   - Shared domain concepts in `domain/`
   - Infrastructure adapters thin and isolated

2. **No business logic in stores or transports**
   - Stores are < 300 lines each
   - MCP tools are < 100 lines each
   - CLI commands are < 50 lines each

3. **Single composition root**
   - `builder.py` constructs all services
   - Config loads once
   - No import-time side effects

4. **One ingestion path**
   - Both session and corpus flow through STM first
   - LTM promotion is reusable pipeline

5. **Enforced boundaries**
   - `tach.toml` rules prevent re-tangling
   - CI fails on boundary violations

6. **Tests stay green**
   - All phases preserve test compatibility
   - No feature regressions

---

## Open Questions

1. **Should graphgen stay as a separate package tree?**
   - Option A: Keep isolated, clearly marked as "batch corpus pipeline"
   - Option B: Dissolve into `memory/ltm/corpus_pipeline/`
   - **Recommendation**: Option A — it's a distinct subsystem with different concerns

2. **What to do with `forget.py`?**
   - Audit callers first
   - If actively used → move to `memory/ltm/forget.py`
   - If stale → remove after deprecation notice

3. **What to do with `migrations/`?**
   - If real feature → build migration runner + registry
   - If admin script → move out of package to `scripts/`

4. **Should `models/` be split into domain vs transport models?**
   - Domain models → `domain/`
   - Transport models (GraphKnowsTool) → `transports/models.py`

5. **How to handle backwards compatibility during transition?**
   - Keep compatibility shims for 1 minor version
   - Document migration guide in CHANGELOG
   - Emit deprecation warnings

---

## Next Steps

Before executing:

1. **Validate this plan with the user**
2. **Choose priority phases** (all 6? or just 0-3?)
3. **Create GitHub issues** for each phase with file touch lists
4. **Run focused tests** to establish baseline
5. **Start with Phase 0** (lowest risk, highest foundation value)

