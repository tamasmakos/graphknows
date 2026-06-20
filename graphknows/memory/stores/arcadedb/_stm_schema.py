"""ArcadeDB STM schema DDL and related constants.

Defines ``_SCHEMA_DDL`` (vertex types, edge types, indexes), ``_VERTEX_TYPE_NAMES``,
``_SINGLE_VALUED_RELATIONS``, and ``_safe_rel_label``. Imported by ``stm_store.py``.
"""

from __future__ import annotations

import re

_DB = "stm"

# Known single-valued relation types — a new fact tombstones the old one.
_SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset(
    {
        "AFFILIATED_WITH",
        "LOCATED_IN",
        "PART_OF",
        "PRODUCED_BY",
        "PRECEDED_BY",
        "SUCCEEDED_BY",
    }
)

# DDL executed in ensure_schema() — order matters (types before indexes).
_SCHEMA_DDL = [
    # Vertex types
    "CREATE VERTEX TYPE SESSION IF NOT EXISTS",
    "CREATE VERTEX TYPE FILE IF NOT EXISTS",
    "CREATE VERTEX TYPE CHUNK IF NOT EXISTS",
    "CREATE VERTEX TYPE ENTITY IF NOT EXISTS",
    "CREATE VERTEX TYPE DIALOG IF NOT EXISTS",
    "CREATE VERTEX TYPE TURN IF NOT EXISTS",
    # Edge types
    "CREATE EDGE TYPE IN_SESSION IF NOT EXISTS",
    "CREATE EDGE TYPE HAS_FILE IF NOT EXISTS",
    "CREATE EDGE TYPE HAS_CHUNK IF NOT EXISTS",
    "CREATE EDGE TYPE HAS_ENTITY IF NOT EXISTS",
    "CREATE EDGE TYPE HAS_TURN IF NOT EXISTS",
    "CREATE EDGE TYPE MENTIONS IF NOT EXISTS",
    "CREATE EDGE TYPE NEXT IF NOT EXISTS",
    # Indexes — ArcadeDB SQL does not support IF NOT EXISTS for CREATE INDEX;
    # duplicate errors are caught and ignored in ensure_schema().
    "CREATE INDEX ON SESSION(id)",
    "CREATE INDEX ON FILE(id)",
    "CREATE INDEX ON CHUNK(id) UNIQUE",
    "CREATE INDEX ON CHUNK(session_id)",
    "CREATE INDEX ON ENTITY(name) UNIQUE",
    "CREATE INDEX ON ENTITY(session_id)",
    "CREATE INDEX ON DIALOG(id) UNIQUE",
    "CREATE INDEX ON DIALOG(session_id)",
    "CREATE INDEX ON TURN(id) UNIQUE",
    "CREATE INDEX ON TURN(session_id)",
]

_VERTEX_TYPE_NAMES: frozenset[str] = frozenset(
    {"SESSION", "FILE", "CHUNK", "ENTITY", "DIALOG", "TURN", "V", "E"}
)


def _safe_rel_label(raw: str) -> str:
    """Convert a relation string to a valid Cypher relationship type label.

    Prefixes with REL_ to avoid collision with vertex type names (ArcadeDB
    requires distinct type registries for vertices and edges).
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    if not label or label[0].isdigit():
        return ""
    if label in _VERTEX_TYPE_NAMES:
        label = f"REL_{label}"
    return label
