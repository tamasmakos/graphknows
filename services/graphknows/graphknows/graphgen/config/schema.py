"""Schema models for the graph structure."""

from typing import Dict, List

from pydantic import BaseModel, Field


class NodeSchema(BaseModel):
    """Node schema definition for graph validation."""

    label: str
    attributes: List[str] = Field(default_factory=list)
    # "segment", "document", "chunk", "derived"
    source_type: str = "segment"


class EdgeSchema(BaseModel):
    """Edge schema definition for graph validation."""

    source_label: str
    target_label: str
    relation_type: str
    is_hierarchical: bool = False


class GraphSchema(BaseModel):
    """
    Defines the target structure of the knowledge graph.
    """

    nodes: Dict[str, NodeSchema]
    edges: List[EdgeSchema] = Field(default_factory=list)


def get_default_schema() -> GraphSchema:
    """Returns the default schema for the graph.

    Hierarchy (E6 layout)::

        SEGMENT (temporal pipeline-run container)
        └─CONTAINS_DOC─► DOC
                          └─HAS_PASSAGE─► PASSAGE (text subdivision)
                                          └─HAS_CHUNK─► CHUNK
                                                        └─MENTIONS─► ENTITY

    SEGMENT nodes capture pipeline-run provenance (run_id, ingested_at, …).
    PASSAGE nodes are text subdivisions (formerly called SEGMENT).
    """
    return GraphSchema(
        nodes={
            # Temporal pipeline-run container (E6 — new)
            "Segment": NodeSchema(
                label="SEGMENT",
                source_type="temporal",
                attributes=[
                    "segment_id",
                    "ingested_at",
                    "run_id",
                    "git_revision",
                    "config_hash",
                    "pipeline_version",
                    "source_count",
                    "input_dir",
                    "taxonomy_name",
                ],
            ),
            # Text subdivision (formerly SEGMENT, now PASSAGE)
            "Passage": NodeSchema(
                label="PASSAGE",
                source_type="passage",
                attributes=["content", "line_number"],
            ),
            "Doc": NodeSchema(label="DOC", source_type="document", attributes=["filename"]),
            "Chunk": NodeSchema(label="CHUNK", source_type="chunk", attributes=["text"]),
            "Entity": NodeSchema(
                label="ENTITY", source_type="derived", attributes=["name", "type"]
            ),
            "Topic": NodeSchema(
                label="TOPIC", source_type="derived", attributes=["title", "keywords"]
            ),
            "Subtopic": NodeSchema(label="SUBTOPIC", source_type="derived", attributes=["title"]),
        },
        edges=[
            # SEGMENT → DOC (new temporal anchor)
            EdgeSchema(
                source_label="SEGMENT",
                target_label="DOC",
                relation_type="CONTAINS_DOC",
                is_hierarchical=True,
            ),
            # DOC → PASSAGE (renamed from HAS_SEGMENT)
            EdgeSchema(
                source_label="DOC",
                target_label="PASSAGE",
                relation_type="HAS_PASSAGE",
                is_hierarchical=True,
            ),
            EdgeSchema(
                source_label="PASSAGE",
                target_label="CHUNK",
                relation_type="HAS_CHUNK",
                is_hierarchical=True,
            ),
            EdgeSchema(
                source_label="CHUNK",
                target_label="ENTITY",
                relation_type="HAS_ENTITY",
                is_hierarchical=False,
            ),
            EdgeSchema(
                source_label="ENTITY",
                target_label="ENTITY",
                relation_type="RELATION",
                is_hierarchical=False,
            ),
            EdgeSchema(
                source_label="ENTITY",
                target_label="SUBTOPIC",
                relation_type="IN_TOPIC",
                is_hierarchical=True,
            ),
            EdgeSchema(
                source_label="SUBTOPIC",
                target_label="TOPIC",
                relation_type="PARENT_TOPIC",
                is_hierarchical=True,
            ),
        ],
    )
