"""Golden-layer schema DDL for the single-database GraphStore.

Pure data: the ordered statement list ``ensure_schema`` executes. Kept out of
``graph_store.py`` so the store module reads as behaviour rather than 145 lines
of table definitions.
"""

from __future__ import annotations

from graphknows.storage.arcadedb._base import VEC_DIM_TOKEN

_VEC_DIM = VEC_DIM_TOKEN

# ---------------------------------------------------------------------------
# DDL — golden-layer schema (plan §1). Order matters: types before properties,
# properties before indexes. Executed idempotently in ensure_schema().
# ---------------------------------------------------------------------------

_CORE_DDL: list[tuple[str, str]] = [
    # ---- Vertex types -----------------------------------------------------
    ("sql", "CREATE VERTEX TYPE SESSION IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SESSION.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.started_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.consolidated_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.label IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSION(session_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE FILE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FILE.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.media_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FILE(file_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE CHUNK IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CHUNK.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.seq IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.kind IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.ts IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.created_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.memory_class IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.atom_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.entity_names IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY CHUNK.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON CHUNK(chunk_id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(state) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE VERTEX TYPE ENTITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ENTITY.name_norm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.aliases IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY ENTITY.graph_embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON ENTITY(name_norm) UNIQUE"),
    ("sql", "CREATE INDEX ON ENTITY(state) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON ENTITY(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Canonical temporal nodes — one global node per resolved ISO date string.
    ("sql", "CREATE VERTEX TYPE TEMPORAL IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TEMPORAL.date IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON TEMPORAL(date) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE TOPIC IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TOPIC.topic_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON TOPIC(topic_id) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON TOPIC(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Semantic-frame instances (deterministic FrameNet layer). Evidence is the
    # EVOKED_BY edge to the CHUNK — no chunk_id string property.
    ("sql", "CREATE VERTEX TYPE FRAME IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FRAME.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.frame_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.lu IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fes IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fillers IF NOT EXISTS LIST"),
    # The frame's definition text + its embedding, so a chunk's evoked frames are
    # vector-matchable by meaning against a query (see query_frames_by_embedding).
    ("sql", "CREATE PROPERTY FRAME.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON FRAME(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(frame_type) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON FRAME(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Role-filled frame instances (WS/D3 layer). Distinct from the FRAME
    # chunk-tag layer above: one FRAME_INSTANCE = one evoked frame whose CORE
    # frame elements were GLiNER2-filled with entity fillers, linked to those
    # entities by PLAYS_ROLE {role} edges and to its evidence CHUNK by
    # FRAME_EVOKED_IN.
    ("sql", "CREATE VERTEX TYPE FRAME_INSTANCE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.frame IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.match_sim IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.role_confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE INDEX ON FRAME_INSTANCE(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FRAME_INSTANCE(frame) NOTUNIQUE"),
    # Community-quality vertex — one per Louvain community_id, holding the
    # native modularityScore/conductance metrics (see compute_community_quality).
    ("sql", "CREATE VERTEX TYPE COMMUNITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY COMMUNITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.conductance IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY COMMUNITY.internal_edges IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.boundary_edges IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.node_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY COMMUNITY.size IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON COMMUNITY(community_id) UNIQUE"),
    # ---- Edge types -------------------------------------------------------
    ("sql", "CREATE EDGE TYPE PART_OF IF NOT EXISTS"),  # CHUNK -> FILE
    ("sql", "CREATE EDGE TYPE IN_SESSION IF NOT EXISTS"),  # FILE -> SESSION
    ("sql", "CREATE EDGE TYPE MENTIONS IF NOT EXISTS"),  # CHUNK -> ENTITY
    ("sql", "CREATE PROPERTY MENTIONS.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE MENTIONS_DATE IF NOT EXISTS"),  # CHUNK -> TEMPORAL
    ("sql", "CREATE EDGE TYPE HAS_TOPIC IF NOT EXISTS"),  # CHUNK -> TOPIC
    ("sql", "CREATE EDGE TYPE EVOKED_BY IF NOT EXISTS"),  # FRAME -> CHUNK
    ("sql", "CREATE PROPERTY EVOKED_BY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE IN_COMMUNITY IF NOT EXISTS"),  # ENTITY -> COMMUNITY
    ("sql", "CREATE EDGE TYPE PLAYS_ROLE IF NOT EXISTS"),  # FRAME_INSTANCE -> ENTITY
    # Every semantic ENTITY->ENTITY fact, regardless of predicate. The predicate is
    # the `predicate` PROPERTY, not the type name: the vocabulary is open and mostly
    # single-use, so a type-per-predicate minted hundreds of near-empty types (and a
    # CREATE EDGE TYPE DDL round-trip on the ingest hot path) while readers already
    # keyed off r.predicate — the type name was never authoritative, because
    # reserved-word predicates had to be escaped (find -> REL_FIND).
    ("sql", "CREATE EDGE TYPE REL IF NOT EXISTS"),  # ENTITY -> ENTITY
    ("sql", "CREATE PROPERTY REL.predicate IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON REL (predicate) NOTUNIQUE"),
    ("sql", "CREATE PROPERTY PLAYS_ROLE.role IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE FRAME_EVOKED_IN IF NOT EXISTS"),  # FRAME_INSTANCE -> CHUNK
]
