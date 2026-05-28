# Phase 0: Schema Normalization Work Brief

**Date**: 2026-05-26  
**GitHub Issues**: #52, #53, #54, #55  
**Wave**: 0 (Foundation)  

---

## Objective

Fix critical label casing inconsistencies that cause silent query failures in production.

## Problem Statement

**Current state**:
- `graphknows/schema.py` declares canonical labels as **ALL UPPERCASE**
- But actual code uses mixed casing:
  - `:Entity` appears 80 times vs `:ENTITY` 51 times
  - `:Topic` appears 21 times vs `:TOPIC` 1 time
  - `:Chunk`, `:Dialog` also have mixed usage

**Impact**:
- Cypher is case-sensitive
- Query with `:Entity` when nodes are labeled `:ENTITY` → returns 0 rows silently
- This is a **production correctness bug**, not just style

## Audit Results

```
Label Usage Summary:
  :ENTITY   - 51 occurrences (8 files)
  :Entity   - 80 occurrences (6 files) ⚠️
  :TOPIC    - 1 occurrence (1 file)
  :Topic    - 21 occurrences (6 files) ⚠️
  :CHUNK    - 26 occurrences (5 files)
  :Chunk    - 13 occurrences (2 files) ⚠️
  :DIALOG   - 13 occurrences (3 files)
  :Dialog   - 4 occurrences (1 file) ⚠️
  :SESSION  - 53 occurrences (6 files) ✓
  :FILE     - 20 occurrences (4 files) ✓
  :TURN     - 25 occurrences (4 files) ✓
```

**Canonical decision** (from `schema.py`):
```
:SESSION, :FILE, :CHUNK, :ENTITY, :DIALOG, :TURN, :TOPIC
```

All uppercase, no exceptions.

## Files Requiring Changes

### High priority (most violations):

1. **`feature_engineering.py`** (lines 69, 70, 520)
   - Uses `:Entity` and `:Topic`
   - GDS enrichment queries

2. **`retrieval/stores/neo4j.py`** (lines 286, 340, 347, 363, 379)
   - Uses `:Entity` and `:Topic`
   - Core LTM read/write paths

3. **`retrieval/stores/memgraph.py`** (lines 471, 514, 515)
   - Uses `:Entity`
   - Core STM write paths

4. **`forget.py`** (lines 22, 34, 52, 53, 58)
   - Uses `:Entity` and `:Topic`
   - Admin/cleanup paths

5. **`recall.py`** (lines 209)
   - Uses `:Topic`
   - LTM recall paths

6. **`mcp/stdio_server.py`** (lines 245, 610)
   - Uses `:Topic`
   - MCP tool handlers

### Medium priority:

7. **`cli/memory.py`** (line 267)
8. **`graphgen/patch/phases/legacy.py`** (lines 46, 145)
9. **`graphgen/pipeline/entity_relation/ontology.py`** (lines 367, 381)

## Implementation Tasks

### Task 1: Create canonical constants (#53)

**File**: `graphknows/domain/schema.py` (new location)

```python
"""Canonical graph schema constants for STM and LTM."""

# Node labels (all uppercase)
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
REL_HAS_DIALOG = "HAS_DIALOG"
REL_HAS_TURN = "HAS_TURN"
REL_BELONGS_TO = "BELONGS_TO"
REL_CO_OCCURS = "CO_OCCURS"
REL_RELATES_TO = "RELATES_TO"
```

### Task 2: Fix label usage in priority files (#54)

**Pattern**: Replace `:Entity` → `:ENTITY`, `:Topic` → `:TOPIC`, etc.

Example fixes:

```python
# Before (feature_engineering.py:69)
MATCH (n:Entity) RETURN id(n) AS id

# After
MATCH (n:ENTITY) RETURN id(n) AS id
```

```python
# Before (neo4j.py:363)
MATCH (t:Topic {id: $topic_id})

# After  
MATCH (t:TOPIC {id: $topic_id})
```

### Task 3: Add regression tests (#55)

**File**: `tests/domain/test_schema.py` (new)

```python
def test_canonical_labels_are_uppercase():
    """Ensure all canonical labels follow UPPERCASE convention."""
    from graphknows.domain.schema import (
        LABEL_SESSION, LABEL_FILE, LABEL_CHUNK, 
        LABEL_ENTITY, LABEL_DIALOG, LABEL_TURN, LABEL_TOPIC
    )
    
    labels = [LABEL_SESSION, LABEL_FILE, LABEL_CHUNK, 
              LABEL_ENTITY, LABEL_DIALOG, LABEL_TURN, LABEL_TOPIC]
    
    for label in labels:
        assert label.isupper(), f"Label {label} must be uppercase"
        assert label == label.upper()

def test_no_mixed_case_labels_in_codebase():
    """Scan codebase for mixed-case label usage."""
    import re
    from pathlib import Path
    
    # Forbidden patterns
    forbidden = [':Entity', ':Topic', ':Chunk', ':Dialog', ':Session', ':File', ':Turn']
    
    violations = []
    base = Path('graphknows')
    for py_file in base.rglob('*.py'):
        with open(py_file) as f:
            for line_no, line in enumerate(f, 1):
                for pattern in forbidden:
                    if pattern in line and not line.strip().startswith('#'):
                        violations.append(f"{py_file}:{line_no} - {pattern}")
    
    assert not violations, f"Found mixed-case labels:\n" + "\n".join(violations[:10])
```

## Validation Checklist

- [ ] All files audited and label casing fixed
- [ ] New `domain/schema.py` with canonical constants created
- [ ] Old `schema.py` migrated to use domain constants
- [ ] Regression tests added and passing
- [ ] Existing tests still pass
- [ ] Manual smoke test: query Neo4j/Memgraph with canonical labels

## Testing Strategy

1. **Unit tests**: Verify constants are uppercase
2. **Integration tests**: Run existing STM/LTM tests (should still pass)
3. **Smoke test**:
   ```cypher
   // Neo4j
   MATCH (e:ENTITY) RETURN count(e) LIMIT 1
   MATCH (t:TOPIC) RETURN count(t) LIMIT 1
   
   // Memgraph  
   MATCH (e:ENTITY) RETURN count(e) LIMIT 1
   ```

## Risks & Mitigation

**Risk**: Breaking change if external code depends on mixed-case labels
**Mitigation**: This is internal only (no public API surface for raw Cypher)

**Risk**: Tests might be asserting on mixed-case labels
**Mitigation**: Fix tests as part of this work

**Risk**: Neo4j already has nodes with wrong casing
**Mitigation**: Schema constraints use correct casing; nodes should match

## Success Criteria

1. Zero occurrences of `:Entity`, `:Topic`, `:Chunk`, `:Dialog` in codebase
2. All Cypher queries use canonical uppercase labels
3. Regression test enforces this in CI
4. All existing tests pass

## Estimated Effort

- Audit: ✅ Complete (0.5h)
- Create domain/schema.py: 0.5h
- Fix files (9 files, ~20 occurrences): 1h
- Add regression tests: 0.5h
- Validate & test: 0.5h

**Total**: ~2.5 hours

---

## Next Steps

After this work completes, we can proceed to:
- Issue #44-47 (package structure documentation)
- Issue #48-51 (config/builder unification)

This foundational work unblocks all subsequent phases.
