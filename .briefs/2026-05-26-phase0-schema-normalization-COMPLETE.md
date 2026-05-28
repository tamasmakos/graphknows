# Phase 0 Completion Summary: Schema Normalization

**Date**: 2026-05-26  
**GitHub Issues**: #52, #53, #54, #55 (Wave 0)  
**Status**: ✅ **COMPLETE**

---

## What We Fixed

### Critical Bug: Label Casing Inconsistency

**Problem**: Cypher queries used mixed casing (`:Entity`, `:Topic`) while schema constraints used uppercase (`:ENTITY`, `:TOPIC`), causing silent query failures (0 rows returned).

**Solution**: Enforced canonical ALL UPPERCASE convention across the codebase.

---

## Changes Made

### 1. Created Domain Layer (`graphknows/domain/`)

**New files**:
- `graphknows/domain/__init__.py`
- `graphknows/domain/schema.py`

**Exports canonical constants**:
```python
# Node labels (ALL UPPERCASE)
LABEL_SESSION = "SESSION"
LABEL_FILE = "FILE"  
LABEL_CHUNK = "CHUNK"
LABEL_ENTITY = "ENTITY"
LABEL_DIALOG = "DIALOG"
LABEL_TURN = "TURN"
LABEL_TOPIC = "TOPIC"

# Relationship types
REL_HAS_FILE = "HAS_FILE"
REL_HAS_CHUNK = "HAS_CHUNK"
REL_HAS_ENTITY = "HAS_ENTITY"
# ... and more
```

**Enforces convention at import time**:
```python
for label in _ALL_LABELS:
    assert label.isupper(), f"Label {label} must be uppercase"
```

### 2. Updated Root-Level `schema.py`

- Added deprecation notice
- Imports canonical constants from `domain.schema`
- Uses constants in constraint/index DDL (f-strings)

### 3. Fixed Label Usage Across Codebase

**Files fixed** (15 label substitutions):
- `graphknows/feature_engineering.py` — `:Entity` → `:ENTITY`, `:Topic` → `:TOPIC`
- `graphknows/forget.py` — `:Entity` → `:ENTITY`, `:Topic` → `:TOPIC`, `:Chunk` → `:CHUNK`
- `graphknows/recall.py` — `:Topic` → `:TOPIC`
- `graphknows/retrieval/stores/neo4j.py` — `:Entity` → `:ENTITY`, `:Topic` → `:TOPIC`
- `graphknows/retrieval/stores/memgraph.py` — `:Entity` → `:ENTITY`
- `graphknows/mcp/stdio_server.py` — `:Topic` → `:TOPIC`
- `graphknows/cli/memory.py` — `:Topic` → `:TOPIC`
- `graphknows/graphgen/pipeline/entity_relation/ontology.py` — `:Entity` → `:ENTITY`

**Total**: 44 mixed-case violations → 0

### 4. Added Regression Tests

**New test file**: `tests/domain/test_schema.py`

**4 tests added**:
1. `test_canonical_labels_are_uppercase` — Validates constants are uppercase
2. `test_canonical_relationships_are_uppercase` — Validates relationship types
3. `test_no_mixed_case_labels_in_codebase` — Scans codebase for violations
4. `test_schema_ddl_uses_canonical_labels` — Validates schema.py uses constants

**Codebase scan** catches forbidden patterns:
```python
forbidden = [':Entity', ':Topic', ':Chunk', ':Dialog', ...]
```

Fails build if any violations found.

### 5. Fixed Test Assertions

**Updated test expectations** in `tests/retrieval/test_neo4j_store.py`:
- Changed `assert "Entity" in cypher` → `assert ":ENTITY" in cypher`
- Changed `assert "Chunk" in cypher` → `assert ":CHUNK" in cypher`
- Changed `assert any("Entity" in cy ...)` → `assert any(":ENTITY" in cy ...)`

---

## Test Results

```
✅ All domain schema tests pass (4/4)
✅ All existing tests pass (1033 passed, 2 skipped)
✅ Zero mixed-case label violations in codebase
```

---

## Impact

### Before
- 44 mixed-case label violations
- Silent query failures in production
- No enforcement of label conventions
- Tests checking for wrong label casing

### After
- 0 violations (enforced by CI)
- All queries use correct canonical labels
- Import-time validation of constants
- Regression tests prevent future violations

---

## Architecture Foundation Established

This work enables:
- **Issue #44-47**: Package structure documentation (can now reference `domain.schema`)
- **Issue #48-51**: Config/builder unification (shared domain layer exists)
- **Wave 1+**: All subsequent refactoring (clean foundation for service decomposition)

---

## Files Changed

**Created** (3 files):
- `graphknows/domain/__init__.py`
- `graphknows/domain/schema.py`
- `tests/domain/test_schema.py`

**Modified** (10 files):
- `graphknows/schema.py`
- `graphknows/feature_engineering.py`
- `graphknows/forget.py`
- `graphknows/recall.py`
- `graphknows/retrieval/stores/neo4j.py`
- `graphknows/retrieval/stores/memgraph.py`
- `graphknows/mcp/stdio_server.py`
- `graphknows/cli/memory.py`
- `graphknows/graphgen/pipeline/entity_relation/ontology.py`
- `tests/retrieval/test_neo4j_store.py`

**Total lines changed**: ~200 lines

---

## Next Steps

### Immediate (Wave 0 remaining)
- Issue #44: Document target package map
- Issue #45: Define allowed import directions
- Issue #46: Map legacy modules to target owners
- Issue #47: Update tach baseline

### After Wave 0 Complete
- Wave 1: Config/builder unification (#48-51)
- Wave 1: Taxonomy extraction (#56-59)
- Wave 1: Domain models (#60-63)

All enabled by this foundation work.
