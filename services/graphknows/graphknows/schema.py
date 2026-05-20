"""Neo4j schema: constraints, indexes, and migration for LTM nodes."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("graphknows.schema")

CONSTRAINTS: list[str] = [
    # Existing (idempotent)
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id_unique IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.community_id IS UNIQUE",
    # New
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:Dialog) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS FOR (u:Utterance) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
]

INDEXES: list[str] = [
    # Existing (idempotent)
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    # New — full-text for dialog/chunk search
    "CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX utterance_text_idx IF NOT EXISTS FOR (u:Utterance) ON EACH [u.text]",
    # Range indexes for temporal queries
    "CREATE INDEX document_ingested_idx IF NOT EXISTS FOR (d:Document) ON (d.ingested_at)",
    "CREATE INDEX utterance_timestamp_idx IF NOT EXISTS FOR (u:Utterance) ON (u.timestamp)",
    "CREATE INDEX dialog_session_idx IF NOT EXISTS FOR (d:Dialog) ON (d.session_id)",
]


def apply_schema(session: Any) -> dict[str, Any]:
    """Apply all constraints and indexes. Idempotent. Returns counts."""
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
    log.info("apply_schema: %d constraints, %d indexes, %d errors",
             applied["constraints"], applied["indexes"], len(applied["errors"]))
    return applied
