"""WriteBackService — export confirmed session entities back to Neo4j."""

from __future__ import annotations

import logging
from neo4j import GraphDatabase

log = logging.getLogger("agent.session.writeback")


class WriteBackService:
    """Exports high-confidence entities from Memgraph session graphs to Neo4j.

    Uses the same deduplication strategy as graphgen/pipeline/graph_cleaning/resolution.py
    — deduplicate by (name, label) for nodes, (src, tgt, relation_type) for edges.

    Attributes:
        neo4j_uri: Bolt URI for the persistent Neo4j KG.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        """Initialize WriteBackService with Neo4j connection details.

        Args:
            neo4j_uri: Bolt URI for the persistent Neo4j KG.
            neo4j_user: Neo4j username.
            neo4j_password: Neo4j password.
        """
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    async def write_back(
        self, session_id: str, entities: list[dict], min_confidence: float = 0.7
    ) -> dict:
        """Write high-confidence entities from a session back to Neo4j.

        Args:
            session_id: The session whose entities to write back.
            entities: List of entity dicts with name, label, confidence keys.
            min_confidence: Minimum confidence threshold for write-back.

        Returns:
            Dict with keys written (int) and skipped (int).
        """
        high_conf = [e for e in entities if e.get("confidence", 1.0) >= min_confidence]
        written = 0
        with self._driver.session() as s:
            for entity in high_conf:
                s.run(
                    "MERGE (e:Entity {name: $name, label: $label})",
                    name=entity["name"],
                    label=entity["label"],
                )
                written += 1
        skipped = len(entities) - written
        log.info("WriteBack session=%s written=%d skipped=%d", session_id, written, skipped)
        return {"written": written, "skipped": skipped}
