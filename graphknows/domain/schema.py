"""Canonical graph schema constants for STM (Memgraph) and LTM (Neo4j).

This module defines the single source of truth for all node labels and relationship
types used across the graphknows memory graph. All Cypher queries MUST use these
constants to ensure consistency.

**Label Convention**: ALL UPPERCASE (no exceptions)
  - Rationale: Cypher is case-sensitive; mixed casing causes silent query failures
  - Schema constraints are defined with uppercase labels
  - Historical inconsistency (`:Entity` vs `:ENTITY`) caused production bugs

**Usage**:
    from graphknows.domain.schema import LABEL_ENTITY, REL_HAS_ENTITY
    
    query = f"MATCH (e:{LABEL_ENTITY}) WHERE e.name = $name RETURN e"
"""

from __future__ import annotations

# ============================================================================
# Node Labels (STM + LTM)
# ============================================================================

LABEL_SESSION = "SESSION"
"""Session node — represents a conversation or work session."""

LABEL_FILE = "FILE"
"""File node — represents an ingested document or source."""

LABEL_CHUNK = "CHUNK"
"""Chunk node — a text segment from a file, embedded in pgvector."""

LABEL_ENTITY = "ENTITY"
"""Entity node — a named entity extracted by GLiNER2 or similar."""

LABEL_DIALOG = "DIALOG"
"""Dialog node — a conversation thread within a session."""

LABEL_TURN = "TURN"
"""Turn node — a single user/assistant exchange in a dialog."""

LABEL_TOPIC = "TOPIC"
"""Topic node — a community cluster with LLM-generated summary."""

# ============================================================================
# Relationship Types
# ============================================================================

REL_HAS_FILE = "HAS_FILE"
"""SESSION → FILE"""

REL_HAS_CHUNK = "HAS_CHUNK"
"""FILE → CHUNK"""

REL_HAS_ENTITY = "HAS_ENTITY"
"""CHUNK → ENTITY (evidence graph)"""

REL_HAS_DIALOG = "HAS_DIALOG"
"""SESSION → DIALOG"""

REL_HAS_TURN = "HAS_TURN"
"""DIALOG → TURN"""

REL_BELONGS_TO = "BELONGS_TO"
"""ENTITY → TOPIC (community membership)"""

REL_CO_OCCURS = "CO_OCCURS"
"""ENTITY → ENTITY (co-occurrence within a chunk)"""

REL_RELATES_TO = "RELATES_TO"
"""ENTITY → ENTITY (semantic relation extracted by NLP)"""

# ============================================================================
# Validation
# ============================================================================

_ALL_LABELS = [
    LABEL_SESSION,
    LABEL_FILE,
    LABEL_CHUNK,
    LABEL_ENTITY,
    LABEL_DIALOG,
    LABEL_TURN,
    LABEL_TOPIC,
]

_ALL_RELS = [
    REL_HAS_FILE,
    REL_HAS_CHUNK,
    REL_HAS_ENTITY,
    REL_HAS_DIALOG,
    REL_HAS_TURN,
    REL_BELONGS_TO,
    REL_CO_OCCURS,
    REL_RELATES_TO,
]

# Enforce uppercase convention at import time
for label in _ALL_LABELS:
    assert label.isupper(), f"Label {label} must be uppercase"

for rel in _ALL_RELS:
    assert rel.isupper(), f"Relationship {rel} must be uppercase"

__all__ = [
    "LABEL_CHUNK",
    "LABEL_DIALOG",
    "LABEL_ENTITY",
    "LABEL_FILE",
    # Node labels
    "LABEL_SESSION",
    "LABEL_TOPIC",
    "LABEL_TURN",
    "REL_BELONGS_TO",
    "REL_CO_OCCURS",
    "REL_HAS_CHUNK",
    "REL_HAS_DIALOG",
    "REL_HAS_ENTITY",
    # Relationship types
    "REL_HAS_FILE",
    "REL_HAS_TURN",
    "REL_RELATES_TO",
]
