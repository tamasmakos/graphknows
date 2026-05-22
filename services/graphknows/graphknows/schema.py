"""Graph schema: constraints and indexes for Neo4j (LTM) and Memgraph (STM)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("graphknows.schema")

# Neo4j LTM constraints
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:SESSION) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:FILE) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:ENTITY) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:DIALOG) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT turn_id_unique IF NOT EXISTS FOR (t:TURN) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (tp:TOPIC) REQUIRE tp.id IS UNIQUE",
]

INDEXES: list[str] = [
    "CREATE INDEX entity_session_idx IF NOT EXISTS FOR (e:ENTITY) ON (e.session_id)",
    "CREATE INDEX chunk_session_idx IF NOT EXISTS FOR (c:CHUNK) ON (c.session_id)",
    "CREATE INDEX turn_session_idx IF NOT EXISTS FOR (t:TURN) ON (t.session_id)",
    "CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:CHUNK) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX turn_text_idx IF NOT EXISTS FOR (t:TURN) ON EACH [t.text]",
]

# Memgraph STM indexes (Memgraph syntax — no IF NOT EXISTS on all)
MEMGRAPH_INDEXES: list[str] = [
    "CREATE INDEX ON :SESSION(id)",
    "CREATE INDEX ON :FILE(id)",
    "CREATE INDEX ON :CHUNK(session_id)",
    "CREATE INDEX ON :ENTITY(session_id)",
    "CREATE INDEX ON :ENTITY(name)",
    "CREATE INDEX ON :DIALOG(session_id)",
    "CREATE INDEX ON :TURN(session_id)",
]


def apply_schema(session: Any) -> dict[str, Any]:
    """Apply Neo4j constraints and indexes. Idempotent. Returns counts."""
    applied: dict[str, Any] = {"constraints": 0, "indexes": 0, "errors": []}
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
            applied["constraints"] += 1
        except Exception as exc:
            applied["errors"].append(f"constraint: {exc}")
    for stmt in INDEXES:
        try:
            session.run(stmt)
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    log.info(
        "apply_schema: %d constraints, %d indexes, %d errors",
        applied["constraints"],
        applied["indexes"],
        len(applied["errors"]),
    )
    return applied


def apply_memgraph_schema(session: Any) -> dict[str, Any]:
    """Apply Memgraph indexes. Idempotent (best-effort). Returns counts."""
    applied: dict[str, Any] = {"indexes": 0, "errors": []}
    for stmt in MEMGRAPH_INDEXES:
        try:
            session.run(stmt).consume()
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    return applied
