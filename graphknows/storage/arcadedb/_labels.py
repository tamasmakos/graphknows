"""Shared label utilities for ArcadeDB stores.

ArcadeDB uses a unified type registry for vertices and edges, so edge labels
must not collide with vertex type names. _safe_rel_label enforces this, and also
escapes SQL reserved keywords so open-vocabulary predicate labels never break
``CREATE EDGE TYPE`` DDL.
"""

from __future__ import annotations

import re

# Union of all vertex type names across STM and LTM databases.
_VERTEX_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "SESSION",
        "FILE",
        "CHUNK",
        "ENTITY",
        "DIALOG",
        "TURN",
        "TOPIC",
        "TEMPORAL",
        "FRAME",
        "FRAME_INSTANCE",
        "V",
        "E",
    }
)

# ArcadeDB SQL reserved keywords that are also common English verbs/nouns, so an
# open-vocabulary predicate can surface them as a bare edge-type name and break
# `CREATE EDGE TYPE <kw>` (e.g. verb "match" -> MATCH, "like" -> LIKE). They get
# the same REL_ escape as vertex-name collisions. Not exhaustive of the grammar —
# just the word-form keywords a natural-language predicate can realistically hit.
_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "MATCH", "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "FROM", "WHERE", "WHILE", "RETURN", "ORDER", "GROUP", "SKIP", "LIMIT",
        "LET", "TRAVERSE", "AND", "OR", "NOT", "NULL", "IN", "IS", "AS", "BY",
        "LIKE", "BETWEEN", "CONTAINS", "DISTINCT", "UNION", "SET", "ADD", "REMOVE",
        "PUT", "MERGE", "MOVE", "FIND", "NEAR", "WITHIN", "LOCK", "WAIT", "RETRY",
        "TIMEOUT", "BEGIN", "COMMIT", "ROLLBACK", "IF", "ELSE", "WITH", "ON", "TO",
        "BOTH", "OUT", "KEY", "VALUE", "VALUES", "DEFAULT", "MAX", "MIN", "ASC",
        "DESC", "BEFORE", "AFTER", "INCREMENT", "DEFINED", "INSTANCEOF", "UNSAFE",
    }
)


def _safe_rel_label(raw: str) -> str:
    """Convert a relation string to a valid Cypher relationship type label.

    Prefixes with REL_ to avoid collision with vertex type names (ArcadeDB
    requires distinct type registries for vertices and edges) and with SQL
    reserved keywords (which cannot be used as a bare type name in DDL).
    Returns an empty string for labels that cannot be made valid.
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    if not label or label[0].isdigit():
        return ""
    if label in _VERTEX_TYPE_NAMES or label in _RESERVED_WORDS:
        label = f"REL_{label}"
    return label
