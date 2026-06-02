"""Canonical graph schema constants for STM (Memgraph) and LTM (Neo4j).

This module defines the single source of truth for all node labels and relationship
types used across the graphknows memory graph. All Cypher queries MUST use these
constants to ensure consistency.

**Label Convention**: ALL UPPERCASE (no exceptions)
  - Rationale: Cypher is case-sensitive; mixed casing causes silent query failures
  - Schema constraints are defined with uppercase labels
  - Historical inconsistency (`:Entity` vs `:ENTITY`) caused production bugs

**Usage**:
    from graphknows.ingestion.domain.schema import LABEL_ENTITY, REL_HAS_ENTITY

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
    # DDL
    "CONSTRAINTS",
    "EdgeSchema",
    "GraphSchema",
    "INDEXES",
    # Node labels
    "LABEL_CHUNK",
    "LABEL_DIALOG",
    "LABEL_DOC",
    "LABEL_ENTITY",
    "LABEL_FILE",
    "LABEL_PASSAGE",
    "LABEL_SEGMENT",
    "LABEL_SESSION",
    "LABEL_SUBTOPIC",
    "LABEL_TOPIC",
    "LABEL_TURN",
    "MEMGRAPH_INDEXES",
    "NodeSchema",
    # Relationship types
    "REL_BELONGS_TO",
    "REL_CO_OCCURS",
    "REL_CONTAINS_DOC",
    "REL_HAS_CHUNK",
    "REL_HAS_DIALOG",
    "REL_HAS_ENTITY",
    "REL_HAS_FILE",
    "REL_HAS_PASSAGE",
    "REL_HAS_TURN",
    "REL_IN_TOPIC",
    "REL_PARENT_TOPIC",
    "REL_RELATES_TO",
    "REL_RELATION",
    # Schema models
    "apply_memgraph_schema",
    "apply_schema",
    "get_default_schema",
]


# ============================================================================
# Additional labels and relationship types for the graphgen corpus pipeline
# (SEGMENT / DOC / PASSAGE hierarchy and topic structure)
# ============================================================================

LABEL_SEGMENT = "SEGMENT"
"""Segment node — provenance anchor for each durable pipeline run."""

LABEL_DOC = "DOC"
"""Doc node — one ingested document within a segment."""

LABEL_PASSAGE = "PASSAGE"
"""Passage node — a logical section within a document."""

LABEL_SUBTOPIC = "SUBTOPIC"
"""Subtopic node — a child topic under a TOPIC."""

REL_CONTAINS_DOC = "CONTAINS_DOC"
"""SEGMENT → DOC"""

REL_HAS_PASSAGE = "HAS_PASSAGE"
"""DOC → PASSAGE"""

REL_RELATION = "RELATION"
"""Generic typed ENTITY → ENTITY relation (typed at the property level)."""

REL_IN_TOPIC = "IN_TOPIC"
"""ENTITY → SUBTOPIC (or TOPIC)"""

REL_PARENT_TOPIC = "PARENT_TOPIC"
"""SUBTOPIC → TOPIC"""


# ============================================================================
# Pydantic schema models (moved from graphgen.config.schema)
# ============================================================================

from pydantic import BaseModel, Field  # noqa: E402 (after constants)


class NodeSchema(BaseModel):
    """Schema definition for a graph node type."""

    label: str = Field(description="Neo4j node label (uppercase)")
    required_properties: list[str] = Field(
        default_factory=list, description="Properties that must always be present"
    )
    optional_properties: list[str] = Field(
        default_factory=list, description="Optional node properties"
    )
    indexed_properties: list[str] = Field(
        default_factory=list, description="Properties with Neo4j indexes"
    )
    unique_constraints: list[str] = Field(
        default_factory=list, description="Properties with UNIQUE constraints"
    )


class EdgeSchema(BaseModel):
    """Schema definition for a graph relationship type."""

    type: str = Field(description="Neo4j relationship type (uppercase)")
    source_label: str = Field(description="Label of the source node")
    target_label: str = Field(description="Label of the target node")
    properties: list[str] = Field(
        default_factory=list, description="Allowed relationship properties"
    )


class GraphSchema(BaseModel):
    """Complete graph schema: collection of node and edge definitions."""

    nodes: list[NodeSchema] = Field(default_factory=list)
    edges: list[EdgeSchema] = Field(default_factory=list)


def get_default_schema() -> GraphSchema:
    """Return the default graphknows graph schema."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label=LABEL_SEGMENT,
                required_properties=["segment_id"],
                optional_properties=["ingested_at", "run_id", "git_revision"],
                unique_constraints=["segment_id"],
            ),
            NodeSchema(
                label=LABEL_DOC,
                required_properties=["id"],
                optional_properties=["title", "source", "doc_type"],
                indexed_properties=["id"],
                unique_constraints=["id"],
            ),
            NodeSchema(
                label=LABEL_PASSAGE,
                required_properties=["id"],
                optional_properties=["heading", "passage_index"],
                indexed_properties=["id"],
            ),
            NodeSchema(
                label=LABEL_CHUNK,
                required_properties=["id"],
                optional_properties=["text", "chunk_index", "heading_path"],
                indexed_properties=["id"],
                unique_constraints=["id"],
            ),
            NodeSchema(
                label=LABEL_ENTITY,
                required_properties=["name"],
                optional_properties=["type", "confidence", "embedding"],
                indexed_properties=["name"],
                unique_constraints=["name"],
            ),
            NodeSchema(
                label=LABEL_TOPIC,
                required_properties=["id"],
                optional_properties=["label", "summary", "community_id"],
                unique_constraints=["id"],
            ),
            NodeSchema(
                label=LABEL_SUBTOPIC,
                required_properties=["id"],
                optional_properties=["label", "summary"],
                unique_constraints=["id"],
            ),
        ],
        edges=[
            EdgeSchema(
                type=REL_CONTAINS_DOC,
                source_label=LABEL_SEGMENT,
                target_label=LABEL_DOC,
            ),
            EdgeSchema(
                type=REL_HAS_PASSAGE,
                source_label=LABEL_DOC,
                target_label=LABEL_PASSAGE,
            ),
            EdgeSchema(
                type=REL_HAS_CHUNK,
                source_label=LABEL_PASSAGE,
                target_label=LABEL_CHUNK,
            ),
            EdgeSchema(
                type=REL_HAS_ENTITY,
                source_label=LABEL_CHUNK,
                target_label=LABEL_ENTITY,
                properties=["confidence"],
            ),
            EdgeSchema(
                type=REL_RELATION,
                source_label=LABEL_ENTITY,
                target_label=LABEL_ENTITY,
                properties=["relation_type", "confidence", "evidence"],
            ),
            EdgeSchema(
                type=REL_IN_TOPIC,
                source_label=LABEL_ENTITY,
                target_label=LABEL_SUBTOPIC,
            ),
            EdgeSchema(
                type=REL_PARENT_TOPIC,
                source_label=LABEL_SUBTOPIC,
                target_label=LABEL_TOPIC,
            ),
        ],
    )


# ============================================================================
# DDL — Neo4j (LTM) constraints/indexes and Memgraph (STM) indexes
# (moved from graphknows.schema)
# ============================================================================

import logging as _logging  # noqa: E402

_log = _logging.getLogger("graphknows.ingestion.domain.schema")

CONSTRAINTS: list[str] = [
    f"CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:{LABEL_SESSION}) REQUIRE s.id IS UNIQUE",
    f"CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:{LABEL_FILE}) REQUIRE f.id IS UNIQUE",
    f"CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:{LABEL_CHUNK}) REQUIRE c.id IS UNIQUE",
    f"CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:{LABEL_ENTITY}) REQUIRE e.name IS UNIQUE",
    f"CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:{LABEL_DIALOG}) REQUIRE d.id IS UNIQUE",
    f"CREATE CONSTRAINT turn_id_unique IF NOT EXISTS FOR (t:{LABEL_TURN}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (tp:{LABEL_TOPIC}) REQUIRE tp.id IS UNIQUE",
]

INDEXES: list[str] = [
    f"CREATE INDEX entity_session_idx IF NOT EXISTS FOR (e:{LABEL_ENTITY}) ON (e.session_id)",
    f"CREATE INDEX chunk_session_idx IF NOT EXISTS FOR (c:{LABEL_CHUNK}) ON (c.session_id)",
    f"CREATE INDEX turn_session_idx IF NOT EXISTS FOR (t:{LABEL_TURN}) ON (t.session_id)",
    f"CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:{LABEL_CHUNK}) ON EACH [c.text]",
    f"CREATE FULLTEXT INDEX turn_text_idx IF NOT EXISTS FOR (t:{LABEL_TURN}) ON EACH [t.text]",
]

MEMGRAPH_INDEXES: list[str] = [
    f"CREATE INDEX ON :{LABEL_SESSION}(id)",
    f"CREATE INDEX ON :{LABEL_FILE}(id)",
    f"CREATE INDEX ON :{LABEL_CHUNK}(session_id)",
    f"CREATE INDEX ON :{LABEL_ENTITY}(session_id)",
    f"CREATE INDEX ON :{LABEL_ENTITY}(name)",
    f"CREATE INDEX ON :{LABEL_DIALOG}(session_id)",
    f"CREATE INDEX ON :{LABEL_TURN}(session_id)",
]


def apply_schema(session) -> dict:
    """Apply Neo4j constraints and indexes. Idempotent. Returns counts."""
    applied: dict = {"constraints": 0, "indexes": 0, "errors": []}
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
            applied["constraints"] += 1
        except Exception as exc:
            applied["errors"].append(f"constraint: {exc}")
    for stmt in INDEXES:
        try:
            session.run(stmt)
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    _log.info(
        "apply_schema: %d constraints, %d indexes, %d errors",
        applied["constraints"],
        applied["indexes"],
        len(applied["errors"]),
    )
    return applied


def apply_memgraph_schema(session) -> dict:
    """Apply Memgraph indexes. Idempotent (best-effort). Returns counts."""
    applied: dict = {"indexes": 0, "errors": []}
    for stmt in MEMGRAPH_INDEXES:
        try:
            session.run(stmt).consume()
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    return applied
