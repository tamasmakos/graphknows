"""
Topics plugin — community detection via the Leiden algorithm.

Wraps the existing CommunityDetector and annotates each Entity node
with a ``community_id`` property that can later be used for
community-level summarisation.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from kg.plugins import GraphPlugin

logger = logging.getLogger(__name__)


class TopicsPlugin(GraphPlugin):
    """Detect entity communities and annotate nodes with community_id."""

    name = "topics"

    async def run(self, graph: nx.DiGraph, **kwargs: Any) -> nx.DiGraph:
        try:
            from kg.community.detection import CommunityDetector
        except ImportError:
            logger.warning("CommunityDetector unavailable — skipping topics plugin")
            return graph

        detector = CommunityDetector()

        # Build entity-only subgraph for community detection
        entity_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("node_type", "").upper() in {"ENTITY", "ENTITY_CONCEPT"}
        ]
        subgraph = graph.subgraph(entity_nodes).copy()

        if subgraph.number_of_nodes() < 3:
            logger.info("Too few entity nodes for community detection — skipping")
            return graph

        community_map: dict[Any, int] = detector.run_leiden(subgraph)

        assigned = 0
        for node_id, community_id in community_map.items():
            if node_id in graph.nodes:
                graph.nodes[node_id]["community_id"] = str(community_id)
                assigned += 1

        logger.info("Topics plugin: assigned community_id to %d nodes", assigned)
        return graph
