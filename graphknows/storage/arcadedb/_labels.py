"""Shared label utilities for ArcadeDB stores.

ArcadeDB uses a unified type registry for vertices and edges, so edge labels
must not collide with vertex type names. _safe_rel_label enforces this.
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
        "SEGMENT",
        "DOC",
        "SESSIONENTITY",
        "V",
        "E",
    }
)


def _safe_rel_label(raw: str) -> str:
    """Convert a relation string to a valid Cypher relationship type label.

    Prefixes with REL_ to avoid collision with vertex type names (ArcadeDB
    requires distinct type registries for vertices and edges).
    Returns an empty string for labels that cannot be made valid.
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    if not label or label[0].isdigit():
        return ""
    if label in _VERTEX_TYPE_NAMES:
        label = f"REL_{label}"
    return label
