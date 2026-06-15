"""Shared helpers for writing entity nodes + edges + relation edges to Memgraph.

Both _STMIngestHandler and DialogCapture follow the same write pattern but use
different edge types (chunk vs. turn).  The edge_writer callable is the only
difference — callers supply a lambda that calls the appropriate edge method.
"""

from __future__ import annotations

from typing import Any, Callable


def write_entity_nodes_and_edges(
    mg: Any,
    session_id: str,
    entities: list[dict[str, Any]],
    edge_writer: Callable[[dict[str, Any]], None],
    _run: Callable,
) -> int:
    """Write entity nodes and their source edge. Returns the entity count.

    Args:
        mg: MemgraphStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        entities: List of entity dicts with "name", optional "type" and "score".
        edge_writer: Callable(ent_dict) that writes the edge linking the entity
            to its source node (chunk or turn).  Callers supply this to avoid
            coupling the helper to a specific edge method.
        _run: Synchronous bridge for async Memgraph calls.
    """
    entity_count = 0
    for ent in entities:
        _run(
            mg.write_entity_node(
                session_id=session_id,
                entity_name=ent["name"],
                entity_type=ent.get("type", "ENTITY"),
                confidence=ent.get("score", 1.0),
            )
        )
        edge_writer(ent)
        entity_count += 1
    return entity_count


def write_relation_edges(
    mg: Any,
    session_id: str,
    relations: list[dict[str, Any]],
    _run: Callable,
) -> None:
    """Write relation edges from extraction results.

    Args:
        mg: MemgraphStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        relations: List of relation dicts with "head", "relation", "tail", optional "score".
        _run: Synchronous bridge for async Memgraph calls.
    """
    for rel in relations:
        _run(
            mg.write_relation_edge(
                session_id=session_id,
                subject_name=rel["head"],
                relation_name=rel["relation"],
                object_name=rel["tail"],
                confidence=rel.get("score", 0.5),
            )
        )
