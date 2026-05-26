"""Graph schema: constraints and indexes for Neo4j (LTM) and Memgraph (STM).

.. deprecated::
   This module will be merged into graphknows.domain.schema in a future version.
   The constraint/index DDL definitions will remain here, but the canonical label
   constants are now in graphknows.domain.schema.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.domain.schema import (
    LABEL_CHUNK,
    LABEL_DIALOG,
    LABEL_ENTITY,
    LABEL_FILE,
    LABEL_SESSION,
    LABEL_TOPIC,
    LABEL_TURN,
)

log = logging.getLogger("graphknows.schema")

# Neo4j LTM constraints (using canonical labels from domain.schema)
CONSTRAINTS: list[str] = [
    f"CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:{LABEL_SESSION}) REQUIRE s.id IS UNIQUE",
    f"CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:{LABEL_FILE}) REQUIRE f.id IS UNIQUE",
    f"CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:{LABEL_CHUNK}) REQUIRE c.id IS UNIQUE",
    f"CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:{LABEL_ENTITY}) REQUIRE e.name IS UNIQUE",
    f"CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:{LABEL_DIALOG}) REQUIRE d.id IS UNIQUE",
    f"CREATE CONSTRAINT turn_id_unique IF NOT EXISTS FOR (t:{LABEL_TURN}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (tp:{LABEL_TOPIC}) REQUIRE tp.id IS UNIQUE",
]

INDEXES: list[str] = [
    f"CREATE INDEX entity_session_idx IF NOT EXISTS FOR (e:{LABEL_ENTITY}) ON (e.session_id)",
    f"CREATE INDEX chunk_session_idx IF NOT EXISTS FOR (c:{LABEL_CHUNK}) ON (c.session_id)",
    f"CREATE INDEX turn_session_idx IF NOT EXISTS FOR (t:{LABEL_TURN}) ON (t.session_id)",
    f"CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:{LABEL_CHUNK}) ON EACH [c.text]",
    f"CREATE FULLTEXT INDEX turn_text_idx IF NOT EXISTS FOR (t:{LABEL_TURN}) ON EACH [t.text]",
]

# Memgraph STM indexes (Memgraph syntax — no IF NOT EXISTS on all)
MEMGRAPH_INDEXES: list[str] = [
    f"CREATE INDEX ON :{LABEL_SESSION}(id)",
    f"CREATE INDEX ON :{LABEL_FILE}(id)",
    f"CREATE INDEX ON :{LABEL_CHUNK}(session_id)",
    f"CREATE INDEX ON :{LABEL_ENTITY}(session_id)",
    f"CREATE INDEX ON :{LABEL_ENTITY}(name)",
    f"CREATE INDEX ON :{LABEL_DIALOG}(session_id)",
    f"CREATE INDEX ON :{LABEL_TURN}(session_id)",
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
