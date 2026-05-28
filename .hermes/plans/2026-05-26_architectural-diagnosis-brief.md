# graphknows Architecture Diagnosis — Brief

**Model**: Claude Sonnet 4.5  
**Date**: 2026-05-26  

---

## The Core Problem

The repo contains **three overlapping architectures**:
1. Legacy HybridRAG container (batch retrieval patterns)
2. New STM/LTM memory service (real-time agent memory)
3. Vendored graphgen (batch corpus pipeline)

This creates confused boundaries, leaky abstractions, and misplaced modules.

---

## Your Intuitions — Validation

| Your complaint | Verdict | Reframing |
|----------------|---------|-----------|
| `feature_engineering.py` at root | ✅ **Correct** | Belongs in `memory/ltm/enrichment.py` |
| `recall.py` / `forget.py` at root | ✅ **Correct** | Belong in `memory/ltm/` |
| `mcp/stdio_server.py` is too large | ✅ **Correct** | 797 lines, needs decomposition |
| Taxonomy in `graphgen/config` | ✅ **Correct** | Shared domain concept, should be `domain/taxonomy.py` |
| `migrations/` is misplaced | ✅ **Correct** | Either build migration system or move to `scripts/` |
| `schema.py` at root | ⚠️ **Partially** | Not the placement issue — the **label casing inconsistency** is the real problem |
| Graphgen not integrated | ⚠️ **Reframe** | It **is** functionally integrated, but architecturally second-class. Should be **explicit first-class batch corpus path** |
| Analytics should be in memory doctor | ❌ **No** | `graphgen/analytics` are batch-oriented; don't conflate with runtime health checks |

---

## Critical Bugs Found

1. **Taxonomy override is broken**
   - MCP `corpus_ingest` sets `config["taxonomy_file"]` (top-level)
   - But `PipelineSettings` expects `settings.extraction.taxonomy_file` (nested)
   - **Result**: Taxonomy override silently ignored

2. **Label casing breaks queries**
   - `:Entity` vs `:ENTITY` in Cypher
   - Inconsistent across codebase
   - **Result**: 0 rows silently returned on wrong casing

3. **GDS enrichment session_id is unused**
   - API signature: `run_gds_enrichment(session_id)`
   - Implementation: enriches **all LTM entities**, ignores session_id
   - **Result**: Semantic mismatch

---

## Target Architecture

```
graphknows/
├── memory/
│   ├── stm/          # Short-term memory service
│   ├── ltm/          # Long-term memory service
│   │   ├── promotion.py   # STM→LTM flush
│   │   ├── enrichment.py  # GDS, Node2Vec
│   │   ├── recall.py
│   │   └── forget.py
│   └── semantic/     # Topics, embeddings
│
├── domain/           # Shared concepts
│   ├── taxonomy.py
│   ├── entities.py
│   └── schema.py     # Canonical labels
│
├── infrastructure/   # Thin adapters
│   ├── neo4j.py
│   ├── memgraph.py
│   ├── pgvector.py
│   └── embedder.py
│
├── transports/       # Thin protocol layers
│   ├── mcp/
│   │   └── tools/    # stm_tools, ltm_tools, diagnostics
│   └── cli/
│
├── graphgen/         # Batch corpus pipeline (first-class)
│   └── ...
│
├── config.py
├── builder.py        # Composition root
└── __init__.py
```

**Key principle**: Repo structure mirrors the service.

---

## Structural Issues Ranked by Impact

### High Impact (breaks semantics or usability)

1. **MCP monolith** (797 lines, hard to test)
2. **Taxonomy boundary leak** (shared domain hidden in graphgen config + broken override)
3. **Label casing inconsistency** (silent query failures)
4. **Stores contain business logic** (neo4j.py 1000+ lines, mixes infra + service logic)

### Medium Impact (confusing but functional)

5. **Misplaced enrichment/recall/forget** (root level instead of `memory/ltm/`)
6. **Duplicated diagnostics** (MCP, CLI, container all have separate health checks)
7. **Scattered config/env loading** (import-time side effects in MCP)
8. **Two disconnected ingestion paths** (session vs corpus)

### Low Impact (aesthetic or technical debt)

9. **Legacy container surface** (stale methods, old pgvector table assumptions)
10. **Stale models** (reflect old pipeline shapes, not service boundaries)
11. **Orphaned migrations** (no runner, unclear if supported)
12. **Graphgen architectural asymmetry** (feels bolted-on vs native)

---

## Refactoring Phases (6 total)

### Phase 0: Normalize foundations ✅ **Low risk, high value**
- Canonical label casing in `domain/schema.py`
- Fix taxonomy override wiring
- Audit Cypher queries for label bugs

### Phase 1: Extract services from stores
- Create `infrastructure/` with thin adapters
- Move business logic to `memory/stm/`, `memory/ltm/`
- Separate enrichment, recall, forget

### Phase 2: Rebuild composition root
- Create `builder.py` (service construction)
- Remove import-time side effects
- Shared config loader

### Phase 3: Decompose MCP monolith
- `transports/mcp/tools/{stm,ltm,corpus,diagnostics}_tools.py`
- Each < 100 lines, unit-testable

### Phase 4: Unify ingestion paths
- Batch corpus flows through STM first
- Reusable LTM promotion pipeline
- Both session + corpus use same enrichment path

### Phase 5: Refactor CLI to reuse services
- CLI becomes thin wrapper over service methods
- Remove duplicated business logic

### Phase 6: Remove legacy surface
- Classify/move `forget.py`, `migrations/`
- Remove `mcp_ltm.py` shim
- Enforce boundaries with `tach.toml` in CI

---

## Recommended Execution Order

**If doing everything**: 0 → 1 → 2 → 3 → 4 → 5 → 6

**If prioritizing correctness**: 0 (fix bugs) → 1 (clean boundaries) → 2 (single composition root)

**If prioritizing maintainability**: 0 → 3 (decompose MCP) → 1 (separate services)

---

## Open Decisions

1. **Graphgen placement**
   - Keep as separate package tree (marked as batch corpus pipeline)?
   - Or dissolve into `memory/ltm/corpus_pipeline/`?
   - **Recommendation**: Keep separate — it's a distinct subsystem

2. **Forget.py status**
   - Audit callers → if active, move to `memory/ltm/forget.py`
   - If stale → remove after deprecation

3. **Migrations status**
   - Build real migration system?
   - Or move to `scripts/` as admin tools?

4. **Backwards compatibility**
   - Keep shims for 1 minor version?
   - Emit deprecation warnings?

---

## Success Criteria

After refactor:

1. Repo structure mirrors service (✅ `memory/stm`, `memory/ltm`, `memory/semantic`)
2. No business logic in stores/transports (✅ all < 300 lines)
3. Single composition root (✅ `builder.py`)
4. One ingestion path (✅ corpus flows through STM)
5. Enforced boundaries (✅ `tach.toml` in CI)
6. All tests green (✅ no regressions)

---

## What I Wrote

1. **Full plan**: `/root/graphknows/.hermes/plans/2026-05-26_architectural-cleanup-plan.md`
2. **This brief**: `/root/graphknows/.hermes/plans/2026-05-26_architectural-diagnosis-brief.md`

**Next**: You decide — execute all phases? Just Phase 0? Create GitHub issues?
