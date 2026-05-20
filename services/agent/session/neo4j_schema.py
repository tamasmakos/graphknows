"""Neo4j LTM schema management — ensures constraints and indexes for Segment data model."""

from __future__ import annotations

import logging

from neo4j import GraphDatabase

log = logging.getLogger("agent.session.neo4j_schema")


SCHEMA_STATEMENTS: list[str] = [
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id_unique IF NOT EXISTS FOR (s:Segment) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT topic_community_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.community_id IS UNIQUE",
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_community_idx IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
    "CREATE INDEX segment_timestamp_idx IF NOT EXISTS FOR (s:Segment) ON (s.timestamp)",
]


def ensure_ltm_schema(neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> list[dict[str, str]]:
    """Run all LTM schema statements (constraints + indexes) and return results.

    Each result dict contains ``{"statement": ..., "status": "created" | "skipped" | "error"}``.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    results: list[dict[str, str]] = []
    try:
        with driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                try:
                    session.run(stmt).consume()
                    log.info("Schema OK: %s", stmt[:80])
                    results.append({"statement": stmt, "status": "created"})
                except Exception as exc:
                    msg = str(exc)
                    if "already exists" in msg or "equivalent constraint already exists" in msg:
                        log.info("Schema skipped (already exists): %s", stmt[:80])
                        results.append({"statement": stmt, "status": "skipped"})
                    else:
                        log.error("Schema error: %s — %s", stmt[:80], msg)
                        results.append({"statement": stmt, "status": "error", "detail": msg})
    finally:
        driver.close()
    return results
