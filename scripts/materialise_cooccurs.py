#!/usr/bin/env python3
"""Materialise CO_OCCURS and DISCUSSES edges in the Graphknows Neo4j KG."""
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run() -> None:
    """Connect to Neo4j and materialise all derived edge types + fix Chunk metadata."""
    driver = GraphDatabase.driver("bolt://172.18.0.1:7687", auth=("neo4j", "testpassword123"))
    with driver.session() as session:
        r = session.run("""
            MATCH (a:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(b:Entity)
            WHERE id(a) < id(b)
            WITH a, b, collect(c.id) AS chunks, count(c) AS weight
            MERGE (a)-[r:CO_OCCURS]->(b)
            SET r.via_chunks = chunks, r.weight = weight
            RETURN count(r) AS n
        """)
        log.info("CO_OCCURS edges: %d", r.single()["n"])

        r2 = session.run("""
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[:IN_COMMUNITY]->(t:Topic)
            MERGE (c)-[r:DISCUSSES]->(t)
            RETURN count(r) AS n
        """)
        log.info("DISCUSSES edges: %d", r2.single()["n"])

        session.run("""
            MATCH (c:Chunk) WHERE c.doc_id = 'ep_debate' OR c.doc_id IS NULL
            SET c.doc_id = 'ep_debate_2024_03_13',
                c.institution = 'European Parliament',
                c.url = 'https://europarl.europa.eu/plenary/en/debates.html',
                c.date = '2024-03-13'
        """)
        log.info("Chunk metadata fixed")

        counts = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n").data()
        log.info("Final edge counts: %s", counts)

    driver.close()


if __name__ == "__main__":
    run()
