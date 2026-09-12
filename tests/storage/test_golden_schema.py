"""The golden-layer DDL is exactly the schema contract, and nothing it retired.

The shape under test is contracts/storage-schema.md. A property written by any
path and absent from ``_CORE_DDL`` is the S6 defect this file exists to prevent;
a type from the retired schema still declared is the dead plane it exists to
prevent.
"""

from __future__ import annotations

import re

from graphknows.storage.arcadedb._schema import _CORE_DDL, schema_version

_STATEMENTS = [stmt for _lang, stmt in _CORE_DDL]

GOLDEN_VERTEX_TYPES = {
    "SCHEMA_STAMP",
    "SOURCE",
    "SEGMENT",
    "ENTITY",
    "CONCEPT",
    "PREDICATE",
    "FACT",
}
GOLDEN_EDGE_TYPES = {
    "PART_OF",
    "NEXT",
    "MENTIONS",
    "EVOKES",
    "INSTANCE_OF",
    "BROADER",
    "MERGED_INTO",
    "SAME_AS_CANDIDATE",
    "SUBJECT_OF",
    "OBJECT_IS",
    "USES",
    "ASSERTED_IN",
    "CITES",
    "SUPERSEDES",
    "CONTRADICTS",
}
GOLDEN_PROPERTIES = {
    "SCHEMA_STAMP": {"stamp_id", "version"},
    "SOURCE": {"id", "uri", "mime", "content_hash", "imported_at", "namespace", "meta"},
    "SEGMENT": {
        "id",
        "source_id",
        "text",
        "kind",
        "path",
        "byte_start",
        "byte_end",
        "role",
        "observed_at",
        "observed_at_inferred",
        "embedding",
        "extractor_version",
        "meta",
    },
    "ENTITY": {
        "id",
        "name",
        "name_norm",
        "type_histogram",
        "block_key",
        "embedding",
        "state",
        "created_at",
    },
    "CONCEPT": {"uri", "label", "definition", "embedding", "pack"},
    "PREDICATE": {
        "id",
        "label",
        "definition",
        "canonical",
        "functional",
        "domain",
        "range",
        "pack",
    },
    "FACT": {
        "id",
        "subject",
        "predicate",
        "object",
        "polarity",
        "modality",
        "confidence",
        "valid_from",
        "valid_to",
        "is_current",
        "extractor",
        "extractor_version",
        "state",
    },
    "MENTIONS": {"surface", "span", "confidence", "extractor"},
    "EVOKES": {"confidence", "roles"},
    "INSTANCE_OF": {"score", "pack"},
    "MERGED_INTO": {"layer", "evidence", "at"},
    "SAME_AS_CANDIDATE": {"score", "layer", "at"},
    "ASSERTED_IN": {"span"},
}
# FR-046: retired by the golden schema, and not written anywhere either.
RETIRED_TYPES = {
    "CHUNK",
    "EPISODE",
    "EVOKED_BY",
    "EXCLUDES_FE",
    "FE",
    "FILE",
    "FRAME",
    "FRAME_EVOKED_IN",
    "FRAME_INSTANCE",
    "HAS_FE",
    "IN_TOPIC",
    "REL",
    "REQUIRES_FE",
    "SEMTYPE",
    "SESSION",
    "TOPIC",
    "TURN",
}
RETIRED_PROPERTIES = {"ENTITY.pagerank", "ENTITY.community_id", "ENTITY.graph_embedding"}


def _declared_types(keyword: str) -> set[str]:
    pattern = re.compile(rf"CREATE {keyword} TYPE (\w+)")
    return {m.group(1) for stmt in _STATEMENTS if (m := pattern.match(stmt))}


def _declared_properties() -> dict[str, set[str]]:
    pattern = re.compile(r"CREATE PROPERTY (\w+)\.(\w+)")
    declared: dict[str, set[str]] = {}
    for stmt in _STATEMENTS:
        if m := pattern.match(stmt):
            declared.setdefault(m.group(1), set()).add(m.group(2))
    return declared


def test_declared_types_are_exactly_the_golden_schema() -> None:
    assert _declared_types("VERTEX") == GOLDEN_VERTEX_TYPES
    assert _declared_types("EDGE") == GOLDEN_EDGE_TYPES


def test_every_contract_property_is_declared() -> None:
    assert _declared_properties() == GOLDEN_PROPERTIES


def test_retired_types_and_properties_are_gone() -> None:
    """A big-bang cutover, not a schema that keeps its predecessor around."""
    declared = _declared_types("VERTEX") | _declared_types("EDGE")
    assert not declared & RETIRED_TYPES

    joined = "\n".join(_STATEMENTS)
    assert not [prop for prop in RETIRED_PROPERTIES if prop in joined]


def test_identity_and_recall_indexes_exist() -> None:
    """Blocking and ANN both index-backed: one generator alone is a silent recall loss."""
    joined = "\n".join(_STATEMENTS)
    for unique in ("SOURCE(id)", "SOURCE(content_hash)", "SEGMENT(id)", "ENTITY(id)"):
        assert f"CREATE INDEX ON {unique} UNIQUE" in joined
    assert "ON ENTITY(block_key) NOTUNIQUE" in joined
    for vector in ("SEGMENT", "ENTITY", "CONCEPT"):
        assert f"ON {vector}(embedding) LSM_VECTOR" in joined


def test_schema_stamp_mechanism_survives_the_replacement() -> None:
    """The singleton index and the dimension-sensitive digest are carried over verbatim."""
    assert "CREATE INDEX ON SCHEMA_STAMP(stamp_id) UNIQUE" in _STATEMENTS
    assert schema_version(768) == schema_version(768)
    assert schema_version(768) != schema_version(1024)
