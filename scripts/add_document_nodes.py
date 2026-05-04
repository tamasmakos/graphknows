#!/usr/bin/env python3
"""Add Document nodes to Neo4j and wire Chunk->Document FROM edges.

Usage::

    PYTHONPATH=/root/graphknows python3 scripts/add_document_nodes.py
"""
from __future__ import annotations

import logging

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
log = logging.getLogger(__name__)

NEO4J_URI = "bolt://172.18.0.1:7687"
NEO4J_AUTH = ("neo4j", "testpassword123")

DOCS: list[dict] = [
    {
        "id": "ep_debate_2024_03_13",
        "title": "EU Strategy in Response to the Ongoing Middle East Crisis",
        "institution": "European Parliament",
        "type": "Plenary Debate",
        "date": "2024-03-13",
        "url": "https://europarl.europa.eu/plenary/en/debates.html",
        "language": "EN",
        "source_id": "ep_debates",
    },
]


def run() -> None:
    """Upsert Document nodes and create FROM edges from matching Chunks."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session() as session:
            for doc in DOCS:
                session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d += $props
                    """,
                    id=doc["id"],
                    props=doc,
                )
                log.info("Upserted Document: %s", doc["id"])

                result = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c)-[r:FROM]->(d)
                    RETURN count(r) AS n
                    """,
                    doc_id=doc["id"],
                )
                edge_count = result.single()["n"]
                log.info("  FROM edges: %d", edge_count)

            # Verification summary
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
            ).data()
            log.info("Node counts: %s", counts)

            from_edges = session.run(
                "MATCH ()-[r:FROM]->() RETURN count(r) AS n"
            ).single()["n"]
            log.info("Total FROM edges: %d", from_edges)
    finally:
        driver.close()


if __name__ == "__main__":
    run()
