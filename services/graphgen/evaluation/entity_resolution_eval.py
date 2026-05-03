from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import networkx as nx

from graphgen.pipeline.graph_cleaning.canonicalization import classify_surface_form


def _entity_nodes(graph: nx.DiGraph) -> List[str]:
    return [
        node_id
        for node_id, node_data in graph.nodes(data=True)
        if node_data.get("node_type") == "ENTITY_CONCEPT"
    ]


def _top_degree_nodes(graph: nx.DiGraph, limit: int = 5) -> List[Dict[str, Any]]:
    entity_nodes = _entity_nodes(graph)
    ranked = sorted(entity_nodes, key=lambda nid: graph.degree(nid), reverse=True)[:limit]
    return [
        {
            "id": node_id,
            "name": graph.nodes[node_id].get("name", node_id),
            "degree": graph.degree(node_id),
        }
        for node_id in ranked
    ]



def summarize_entity_resolution_effects(before: nx.DiGraph, after: nx.DiGraph) -> Dict[str, Any]:
    before_nodes = _entity_nodes(before)
    after_nodes = _entity_nodes(after)

    class_counts = Counter(
        classify_surface_form(after.nodes[node_id].get("name", node_id))
        for node_id in after_nodes
    )

    return {
        "entity_nodes_before": len(before_nodes),
        "entity_nodes_after": len(after_nodes),
        "merged_nodes": max(0, len(before_nodes) - len(after_nodes)),
        "top_degree_nodes_before": _top_degree_nodes(before),
        "top_degree_nodes_after": _top_degree_nodes(after),
        "surface_form_class_counts": dict(class_counts),
    }
