---
title: Ontology Injection Bug & Patch Design
created: 2026-05-08
updated: 2026-05-08
type: concept
tags: [pipeline, neo4j, ontology, patch, architecture, write-mode]
sources: [ontology_bug_report.md]
confidence: high
---

# Ontology Injection Bug & Patch Design

## Executive Summary

Investigation found two classes of issue. The ontology injection bug has a narrower scope than initially suspected, but the remediation required exposes a real architectural gap: no write-mode on the graphgen service.

---

## Findings from Investigation

### ENTITY_CONCEPT nodes: 100% typed 
All 4,018 `ENTITY_CONCEPT` nodes have `ontology_class` set. The current extraction pipeline works.

### Real issues found:

| Category | Volume | Problem |
|---|---|---|
| Structural nodes (CHUNK/SEGMENT/DOC/TOPIC/SUBTOPIC) | 1,698 | No `ontology_class` by design — but creates confusion for queries |
| Legacy orphan nodes (Entity/Chunk/Topic labels) | 143 | Old ingestion shape — no `node_type`, no `id`, wrong schema |
| ENTITY_CONCEPT with `UNKNOWN` class | 84 | CDM ontology couldn't classify these — real quality gap |
| `ontology_type` property | 0 | Never written — stale name expectation vs actual `ontology_class` |

### True ontology injection issues:
1. **84 ENTITY_CONCEPT nodes stuck at `UNKNOWN`** — GLiNER/DSPy returned a label that didn't map to CDM ontology class
2. **Structural nodes** never receive semantic `ontology_class` — downstream queries can't filter entity-type nodes cleanly
3. **Legacy nodes** (143) are schema orphans — wrong label, no `id`, not addressable by current queries
4. **`ontology_type`** was expected by something but never written — likely a consumer mismatch

---

## Architecture Decision: Write-Mode Patch

### Problem
The existing database has 5,859 nodes. Re-running the full graphgen pipeline (OCR → chunking → GLiNER2 → DSPy → graph → Neo4j upload) would cost significant compute and time. We need a targeted write-mode that can:
- Connect to existing Neo4j
- Apply ontology corrections/backfills
- Leave the rest untouched

### Solution: `graphgen patch` subcommand

A new CLI entry point (or service endpoint) that:
1. Reads existing Neo4j nodes
2. Applies ontology classification fixes per node type
3. Writes back to Neo4j in-place (MERGE + SET, no DELETE)
4. Is idempotent — safe to re-run

This becomes a first-class mode alongside the existing pipeline run mode.

---

## Patch Operations (3 phases)

### Phase 1: Structural node classification
**Target:** CHUNK, SEGMENT, DOC, TOPIC, SUBTOPIC nodes (1,698 nodes)
**Strategy:** Deterministic mapping from `node_type` to `ontology_class`

```
CHUNK    → structural_class = "STRUCTURAL_CHUNK"
SEGMENT  → structural_class = "STRUCTURAL_SEGMENT"
DOC      → structural_class = "STRUCTURAL_DOCUMENT"
TOPIC    → structural_class = "TOPIC"
SUBTOPIC → structural_class = "SUBTOPIC"
```

**Cypher pattern:**
```cypher
MATCH (n) WHERE n.node_type = 'CHUNK' AND n.ontology_class IS NULL
SET n.ontology_class = 'STRUCTURAL_CHUNK'
```

### Phase 2: Legacy node schema migration
**Target:** 143 nodes with labels `Entity`, `Chunk`, `Topic` (no `node_type`, no `id`)
**Strategy:**
1. Assign `node_type` from label
2. Generate `id` from `name` (hash or normalize)
3. Classify `ontology_class` from `type` property or name heuristics
4. Upgrade label to current schema (add `ENTITY_CONCEPT` if classifiable)

### Phase 3: UNKNOWN entity reclassification
**Target:** 84 ENTITY_CONCEPT with `ontology_class='UNKNOWN'`
**Strategy:**
1. Rule-based: name patterns (date-like → EVENT, known org suffixes → ORGANIZATION, etc.)
2. LLM-assisted: if rules fail, use Groq/local LLM with CDM ontology context
3. Fallback: keep `UNKNOWN` but set `classification_confidence='low'`

---

## Pipeline Integration (Forward Fix)

### Issue in current forward pipeline
Future runs must also:
1. Assign structural `ontology_class` to CHUNK/SEGMENT/DOC/TOPIC/SUBTOPIC at creation time
2. Improve UNKNOWN fallback: try rule-based classification before giving up
3. Ensure `ontology_type` is not expected anywhere (remove stale references, or add as alias for `ontology_class`)

### Code changes required:
- `lexical_graph_building/builder.py`: assign `ontology_class='STRUCTURAL_CHUNK'` etc. on node creation
- `community/subcommunities.py`: assign `ontology_class='TOPIC'`/`'SUBTOPIC'`
- `entity_relation/extraction.py`: improve UNKNOWN fallback classification
- `utils/schema_utils.py` or new `ontology_classifier.py`: centralized ontology class resolver

---

## Write-Mode Service Design

### API / CLI Interface

```bash
# CLI patch command
graphgen patch --neo4j-uri bolt://172.18.0.1:7687 \
               --phases structural,legacy,unknown \
               --dry-run        # show what would change
               --batch-size 500

# Or as HTTP endpoint (if service-mode)
POST /graphgen/patch
{
  "phases": ["structural", "legacy", "unknown"],
  "dry_run": true,
  "batch_size": 500
}
```

### Implementation location
`services/graphgen/patch/` — new submodule:
```
services/graphgen/patch/
  __init__.py
  runner.py          # orchestrates phases
  phases/
    structural.py    # Phase 1: structural node classification
    legacy.py        # Phase 2: legacy node migration
    unknown.py       # Phase 3: UNKNOWN reclassification
  ontology_classifier.py  # shared rule-based + LLM classifier
  report.py          # patch run summary/diff report
```

### Service integration
The existing `iterative_orchestrator.py` runs a full pipeline. The patch runner is separate — it connects directly to Neo4j without running extraction stages.

---

## Tests Required

### Unit tests
- `test_structural_phase.py`: verify all structural node types get correct class
- `test_legacy_phase.py`: legacy node property migration
- `test_unknown_phase.py`: rule-based classifier for known patterns, LLM fallback
- `test_ontology_classifier.py`: CDM label → ontology_class mapping

### Integration tests
- `test_patch_runner_dry_run.py`: dry-run returns correct diff without writing
- `test_patch_runner_idempotent.py`: running patch twice produces same result
- `test_patch_runner_live_neo4j.py`: validates all nodes have ontology_class post-patch

### Validation
- Post-patch query: `MATCH (n) WHERE n.ontology_class IS NULL RETURN count(n)` → must return 0
- Coverage report: `ontology_class` distribution before/after

---

## Acceptance Criteria

1. `graphgen patch --dry-run` reports all nodes that would be updated with new `ontology_class`
2. `graphgen patch` applied to existing DB leaves 0 nodes with missing `ontology_class`
3. Patch is idempotent (safe to run multiple times)
4. All future pipeline runs (new ingest) produce 0 nodes with missing `ontology_class`
5. Tests pass: unit + integration at ≥80% coverage on patch module
6. `ontology_type` stale references cleaned up or aliased

---

## Related Pages
- [[pipeline-stages]]
- [[knowledge-graph-construction]]
- [[graphgen-architecture]]
