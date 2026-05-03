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
    """
    Returns the strict default schema for the graph.
    Hierarchy:
    (DOC)-[HAS_SEGMENT]-(SEGMENT)-[HAS_CHUNK]-(CHUNK)-[HAS_ENTITY]-(ENTITY)
    (ENTITY)-[RELATION]-(ENTITY)
    (ENTITY)-[IN_TOPIC]-(SUBTOPIC)
    (SUBTOPIC)-[PARENT_TOPIC]-(TOPIC)
    (DOC)-[HAS_TOPIC]-(TOPIC) - Optional shortcut? Keeping minimal for now.
    """
    return GraphSchema(
        nodes={
            "Doc": NodeSchema(label="DOC", source_type="document", attributes=["filename"]),
            "Segment": NodeSchema(label="SEGMENT", source_type="segment", attributes=["content", "line_number"]),
            "Chunk": NodeSchema(label="CHUNK", source_type="chunk", attributes=["text"]),
            "Entity": NodeSchema(label="ENTITY", source_type="derived", attributes=["name", "type"]),
            "Topic": NodeSchema(label="TOPIC", source_type="derived", attributes=["title", "keywords"]),
            "Subtopic": NodeSchema(label="SUBTOPIC", source_type="derived", attributes=["title"]),
        },
        edges=[
            EdgeSchema(source_label="DOC", target_label="SEGMENT", relation_type="HAS_SEGMENT", is_hierarchical=True),
            EdgeSchema(source_label="SEGMENT", target_label="CHUNK", relation_type="HAS_CHUNK", is_hierarchical=True),
            EdgeSchema(source_label="CHUNK", target_label="ENTITY", relation_type="HAS_ENTITY", is_hierarchical=False),
            EdgeSchema(source_label="ENTITY", target_label="ENTITY", relation_type="RELATION", is_hierarchical=False),
            EdgeSchema(source_label="ENTITY", target_label="SUBTOPIC", relation_type="IN_TOPIC", is_hierarchical=True),
            EdgeSchema(source_label="SUBTOPIC", target_label="TOPIC", relation_type="PARENT_TOPIC", is_hierarchical=True),
            # Keeping HAS_TOPIC from Doc if needed for global topics, but user requested hierarchy above.
            # " (DOC)...(TOPIC) " was not explicitly linked in the user ascii art trace, 
            # but usually Docs have topics. The user wrote: "...-(TOPIC)" at the end of chain?
            # Ascii: (ENTITY)-...-(SUBTOPIC)-[PARENT_TOPIC]-(TOPIC)
            # It doesn't show DOC connecting to TOPIC directly.
        ]
    )

