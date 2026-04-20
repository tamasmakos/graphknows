"""
Declarative graph schema for GraphKnows.

Single source of truth: DOCUMENT → CHUNK → ENTITY.
All Cypher should reference these constants rather than hard-coded strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class PropertySpec:
    name: str
    type: str  # "string" | "integer" | "float" | "boolean" | "list[float]"
    required: bool = False
    indexed: bool = False
    unique: bool = False
    vector_dims: int = 0  # > 0 → pgvector / Neo4j vector index


@dataclass(frozen=True)
class NodeSpec:
    label: str
    properties: Sequence[PropertySpec] = field(default_factory=tuple)


@dataclass(frozen=True)
class EdgeSpec:
    type: str
    from_label: str
    to_label: str
    properties: Sequence[PropertySpec] = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphSchema:
    nodes: Sequence[NodeSpec]
    edges: Sequence[EdgeSpec]

    def node(self, label: str) -> NodeSpec:
        for n in self.nodes:
            if n.label == label:
                return n
        raise KeyError(label)

    def edge(self, type_: str) -> EdgeSpec:
        for e in self.edges:
            if e.type == type_:
                return e
        raise KeyError(type_)


# ── Node labels ───────────────────────────────────────────────────────────────
LABEL_DOCUMENT = "Document"
LABEL_CHUNK = "Chunk"
LABEL_ENTITY = "Entity"

# ── Relationship types ────────────────────────────────────────────────────────
REL_CONTAINS = "CONTAINS"  # Document → Chunk
REL_MENTIONS = "MENTIONS"  # Chunk    → Entity
REL_RELATED_TO = "RELATED_TO"  # Entity   → Entity

# ── Canonical schema instance ─────────────────────────────────────────────────
CORE_SCHEMA = GraphSchema(
    nodes=(
        NodeSpec(
            label=LABEL_DOCUMENT,
            properties=(
                PropertySpec("doc_id", "string", required=True, unique=True),
                PropertySpec("title", "string", required=True),
                PropertySpec("source_path", "string"),
                PropertySpec("content_hash", "string", indexed=True),
                PropertySpec("mime_type", "string"),
                PropertySpec("created_at", "string"),
                PropertySpec("chunk_count", "integer"),
            ),
        ),
        NodeSpec(
            label=LABEL_CHUNK,
            properties=(
                PropertySpec("chunk_id", "string", required=True, unique=True),
                PropertySpec("doc_id", "string", required=True, indexed=True),
                PropertySpec("position", "integer", required=True),
                PropertySpec("text", "string", required=True),
                PropertySpec("heading_path", "string"),
                PropertySpec("token_count", "integer"),
                PropertySpec("embedding", "list[float]", vector_dims=384),
            ),
        ),
        NodeSpec(
            label=LABEL_ENTITY,
            properties=(
                PropertySpec("entity_id", "string", required=True, unique=True),
                PropertySpec("name", "string", required=True, indexed=True),
                PropertySpec("type", "string", indexed=True),
                PropertySpec("description", "string"),
                PropertySpec("community_id", "string", indexed=True),
                PropertySpec("embedding", "list[float]", vector_dims=384),
            ),
        ),
    ),
    edges=(
        EdgeSpec(REL_CONTAINS, LABEL_DOCUMENT, LABEL_CHUNK),
        EdgeSpec(
            REL_MENTIONS,
            LABEL_CHUNK,
            LABEL_ENTITY,
            properties=(PropertySpec("weight", "float"),),
        ),
        EdgeSpec(
            REL_RELATED_TO,
            LABEL_ENTITY,
            LABEL_ENTITY,
            properties=(
                PropertySpec("relation_type", "string"),
                PropertySpec("weight", "float"),
                PropertySpec("description", "string"),
            ),
        ),
    ),
)
