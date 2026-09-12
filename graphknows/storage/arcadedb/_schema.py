"""Golden-layer schema DDL for the single-database GraphStore.

Pure data: the ordered statement list ``ensure_schema`` executes. Kept out of
``graph_store.py`` so the store module reads as behaviour rather than a screen
of table definitions.
"""

from __future__ import annotations

import hashlib

from graphknows.storage.arcadedb._base import VEC_DIM_TOKEN

_VEC_DIM = VEC_DIM_TOKEN

# ---------------------------------------------------------------------------
# DDL — golden-layer schema (contracts/storage-schema.md). Order matters: types
# before properties, properties before indexes. Executed idempotently in
# ensure_schema(). Every property any writer sets is declared here: an
# undeclared, load-bearing provenance property is the defect this list exists to
# prevent.
# ---------------------------------------------------------------------------

_CORE_DDL: list[tuple[str, str]] = [
    # ---- Vertex types -----------------------------------------------------
    # Singleton per database, recording the schema shape that created the graph
    # (see schema_version below). The UNIQUE index on stamp_id is what makes it a
    # singleton rather than a convention: a concurrent second insert is rejected
    # by the index, not by a race-prone read-then-write.
    ("sql", "CREATE VERTEX TYPE SCHEMA_STAMP IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SCHEMA_STAMP.stamp_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SCHEMA_STAMP.version IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SCHEMA_STAMP(stamp_id) UNIQUE"),
    # One ingested artefact — a file, a transcript, a session. `content_hash` is
    # UNIQUE as well as `id`: re-importing the same bytes under a new uri must
    # collide rather than fork a second source.
    ("sql", "CREATE VERTEX TYPE SOURCE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SOURCE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.uri IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.mime IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.content_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.imported_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.namespace IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.meta IF NOT EXISTS MAP"),
    ("sql", "CREATE INDEX ON SOURCE(id) UNIQUE"),
    ("sql", "CREATE INDEX ON SOURCE(content_hash) UNIQUE"),
    # The unit of retrieval and of provenance: a turn, a paragraph, a code span.
    # `observed_at_inferred` marks a timestamp derived rather than read, so a
    # temporal answer can say which it stood on.
    ("sql", "CREATE VERTEX TYPE SEGMENT IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SEGMENT.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.source_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.kind IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.path IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.byte_start IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SEGMENT.byte_end IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SEGMENT.role IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.observed_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.observed_at_inferred IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY SEGMENT.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY SEGMENT.extractor_version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.meta IF NOT EXISTS MAP"),
    ("sql", "CREATE INDEX ON SEGMENT(id) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON SEGMENT(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # `block_key` and `embedding` are both indexed because identity resolution
    # runs two candidate generators over them — a blocking key lookup and an ANN
    # probe — and a candidate set built from only one of them is a silent recall
    # loss rather than an error.
    ("sql", "CREATE VERTEX TYPE ENTITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ENTITY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name_norm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type_histogram IF NOT EXISTS MAP"),
    ("sql", "CREATE PROPERTY ENTITY.block_key IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.created_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON ENTITY(id) UNIQUE"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON ENTITY(block_key) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON ENTITY(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # The symbolic index layer: a concept is retrievable by the embedding of its
    # definition, which is what makes the vector index part of the symbol rather
    # than a side-car store.
    ("sql", "CREATE VERTEX TYPE CONCEPT IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CONCEPT.uri IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CONCEPT.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CONCEPT.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CONCEPT.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY CONCEPT.pack IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON CONCEPT(uri) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CONCEPT(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # `functional` is read, not decorative: a functional predicate is what makes
    # a second object for the same subject a contradiction rather than a second
    # fact.
    ("sql", "CREATE VERTEX TYPE PREDICATE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY PREDICATE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.canonical IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.functional IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY PREDICATE.domain IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.range IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY PREDICATE.pack IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON PREDICATE(id) UNIQUE"),
    # A fact is a vertex, not an edge: it carries validity, provenance and a
    # tombstone `state`, and edges cannot be edge endpoints in ArcadeDB.
    ("sql", "CREATE VERTEX TYPE FACT IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FACT.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.subject IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.predicate IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.object IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.polarity IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.modality IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY FACT.valid_from IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.valid_to IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.is_current IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY FACT.extractor IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.extractor_version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FACT.state IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FACT(id) UNIQUE"),
    # ---- Edge types -------------------------------------------------------
    ("sql", "CREATE EDGE TYPE PART_OF IF NOT EXISTS"),  # SEGMENT -> SOURCE
    ("sql", "CREATE EDGE TYPE NEXT IF NOT EXISTS"),  # SEGMENT -> SEGMENT
    # SEGMENT -> ENTITY. `span` locates the surface form inside the segment
    # text, which is what lets a recalled fact quote its own evidence.
    ("sql", "CREATE EDGE TYPE MENTIONS IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY MENTIONS.surface IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY MENTIONS.span IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY MENTIONS.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY MENTIONS.extractor IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE EVOKES IF NOT EXISTS"),  # SEGMENT -> CONCEPT
    ("sql", "CREATE PROPERTY EVOKES.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY EVOKES.roles IF NOT EXISTS LIST"),
    ("sql", "CREATE EDGE TYPE INSTANCE_OF IF NOT EXISTS"),  # ENTITY -> CONCEPT
    ("sql", "CREATE PROPERTY INSTANCE_OF.score IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.pack IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE BROADER IF NOT EXISTS"),  # CONCEPT -> CONCEPT
    # ENTITY -> ENTITY. `evidence` is the load-bearing provenance of a merge —
    # why two mentions became one entity — and is declared for exactly that
    # reason.
    ("sql", "CREATE EDGE TYPE MERGED_INTO IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY MERGED_INTO.layer IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY MERGED_INTO.evidence IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY MERGED_INTO.at IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE SAME_AS_CANDIDATE IF NOT EXISTS"),  # ENTITY -> ENTITY
    ("sql", "CREATE PROPERTY SAME_AS_CANDIDATE.score IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY SAME_AS_CANDIDATE.layer IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SAME_AS_CANDIDATE.at IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE SUBJECT_OF IF NOT EXISTS"),  # ENTITY -> FACT
    ("sql", "CREATE EDGE TYPE OBJECT_IS IF NOT EXISTS"),  # FACT -> ENTITY
    ("sql", "CREATE EDGE TYPE USES IF NOT EXISTS"),  # FACT -> PREDICATE
    ("sql", "CREATE EDGE TYPE ASSERTED_IN IF NOT EXISTS"),  # FACT -> SEGMENT
    ("sql", "CREATE PROPERTY ASSERTED_IN.span IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE CITES IF NOT EXISTS"),  # FACT -> SOURCE
    ("sql", "CREATE EDGE TYPE SUPERSEDES IF NOT EXISTS"),  # FACT -> FACT
    ("sql", "CREATE EDGE TYPE CONTRADICTS IF NOT EXISTS"),  # FACT -> FACT
]


def schema_version(dims: int) -> str:
    """Short digest of the schema shape a graph was built with.

    Derived from the DDL statement text — ``_CORE_DDL`` is
    ``list[tuple[str, str]]``, so the statement is projected out of each pair —
    combined with ``dims``, which the DDL interpolates into the ``LSM_VECTOR``
    indexes: a graph built at one embedding dimension cannot serve another.

    Because the digest is over the text, editing ``_CORE_DDL`` at all — a
    reworded comment inside a statement included — is now a client-visible
    event: every existing graph is stranded and must be rebuilt from its
    original sources. There is no migration.
    """
    ddl_text = "\n".join(stmt for _lang, stmt in _CORE_DDL)
    return hashlib.sha256(f"{ddl_text}\n{dims}".encode()).hexdigest()[:16]
