"""Shared helpers: UUID namespace, node ID generators, label sanitiser."""

from __future__ import annotations

import re
import uuid

# UUID5 namespace for graphknows nodes (fixed sentinel)
_GK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def chunk_node_id(doc_id: str, chunk_hash: str) -> str:
    """Deterministic UUID5 for a Chunk node — idempotent across pipeline runs."""
    return str(uuid.uuid5(_GK_NS, f"chunk:{doc_id}:{chunk_hash}"))


def entity_node_id(name: str) -> str:
    """Deterministic UUID5 for an Entity node keyed on normalised name."""
    return str(uuid.uuid5(_GK_NS, f"entity:{name.strip().upper()}"))


def _safe_rel_label(raw: str) -> str:
    """Convert a free-text relation string to a valid Cypher relationship type.

    Uppercases, replaces spaces/hyphens with underscores, strips non-alphanumeric
    characters. Returns empty string if nothing valid remains.
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    return label
