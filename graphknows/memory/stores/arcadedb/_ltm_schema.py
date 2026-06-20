"""ArcadeDB LTM schema DDL and related constants.

Defines ``_SCHEMA_DDL`` (vertex types, properties, indexes, edge types)
and helpers that operate on label strings. Imported by ``ltm_store.py``.
"""

from __future__ import annotations

import re
import uuid

_LTM_DB = "ltm"
_GK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _entity_node_id(name: str) -> str:
    return str(uuid.uuid5(_GK_NS, f"entity:{name.strip().upper()}"))


_VERTEX_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "SESSION",
        "FILE",
        "CHUNK",
        "ENTITY",
        "DIALOG",
        "TURN",
        "TOPIC",
        "SEGMENT",
        "DOC",
        "SESSIONENTITY",
        "V",
        "E",
    }
)


def _safe_rel_label(raw: str) -> str:
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    if not label or label[0].isdigit():
        return ""
    if label in _VERTEX_TYPE_NAMES:
        label = f"REL_{label}"
    return label


_SCHEMA_DDL: list[tuple[str, str]] = [
    # Vertex types
    ("sql", "CREATE VERTEX TYPE ENTITY"),
    ("sql", "CREATE PROPERTY ENTITY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY ENTITY.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.betweenness IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.degree_centrality IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.ingested_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.contributing_session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.merged_session_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON ENTITY(name) UNIQUE"),
    ("sql", "CREATE INDEX ON ENTITY(id)"),
    ("sql", "CREATE VERTEX TYPE CHUNK"),
    ("sql", "CREATE PROPERTY CHUNK.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.heading_path IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.chunk_index IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON CHUNK(id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id)"),
    ("sql", "CREATE VERTEX TYPE TOPIC"),
    ("sql", "CREATE PROPERTY TOPIC.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.topic_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TOPIC.community_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.entity_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TOPIC.entity_names IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.session_ids IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.gds_modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.community_algorithm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.silhouette IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE INDEX ON TOPIC(id) UNIQUE"),
    ("sql", "CREATE INDEX ON TOPIC(community_id)"),
    ("sql", "CREATE VERTEX TYPE FILE"),
    ("sql", "CREATE PROPERTY FILE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.title IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.parser_name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.source_hint IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FILE(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE DIALOG"),
    ("sql", "CREATE PROPERTY DIALOG.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY DIALOG.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY DIALOG.started_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON DIALOG(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SESSION"),
    ("sql", "CREATE PROPERTY SESSION.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.flushed_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSION(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SEGMENT"),
    ("sql", "CREATE PROPERTY SEGMENT.segment_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.ingested_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.run_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.git_revision IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.config_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.pipeline_version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.source_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SEGMENT.input_dir IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.taxonomy_name IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SEGMENT(segment_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE DOC"),
    ("sql", "CREATE PROPERTY DOC.id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON DOC(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SESSIONENTITY"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSIONENTITY(session_id)"),
    # Edge types
    ("sql", "CREATE EDGE TYPE MENTIONS"),
    ("sql", "CREATE EDGE TYPE HAS_CHUNK"),
    ("sql", "CREATE EDGE TYPE IN_SESSION"),
    ("sql", "CREATE EDGE TYPE HAS_TURN"),
    ("sql", "CREATE EDGE TYPE NEXT"),
    ("sql", "CREATE EDGE TYPE IN_TOPIC"),
    ("sql", "CREATE EDGE TYPE IN_COMMUNITY"),
    ("sql", "CREATE EDGE TYPE CO_OCCURS"),
    ("sql", "CREATE EDGE TYPE APPEARED_IN"),
    ("sql", "CREATE EDGE TYPE CONTAINS_DOC"),
    ("sql", "CREATE EDGE TYPE HAS_ENTITY"),
]
