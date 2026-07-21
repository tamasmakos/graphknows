"""Shared SQL/Cypher string helpers for ArcadeDB stores.

ArcadeDB's openCypher dialect does not support ``IN $list`` / ``any(x IN $list)``
parameter binding, so list membership and name matching are built by string
interpolation. Every interpolated string value MUST pass through these helpers —
they are the single audited home for quote-stripping that was previously
copy-pasted across stores and channels.
"""

from __future__ import annotations

from collections.abc import Iterable


def sanitize(value: str) -> str:
    """Escape *value* for embedding inside a single-quoted literal.

    Backslash must be escaped first, otherwise a value ending in one would
    escape the closing quote and let the rest of the value be parsed as query
    text. Quotes are escaped rather than stripped: entity names are *written*
    verbatim through bound parameters, so stripping made every name containing
    an apostrophe ("O'Brien") impossible to look up again.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def quote(value: str) -> str:
    """Return *value* as a sanitized single-quoted SQL string literal."""
    return f"'{sanitize(value)}'"


def quoted_list(values: Iterable[str]) -> str:
    """Return values as a Cypher/SQL list literal: ``['a','b']``."""
    return "[" + ",".join(quote(v) for v in values) + "]"


def is_missing_type(exc: Exception) -> bool:
    """True when an error only means the vertex type does not exist yet.

    A database that never ingested a given label has no such type, so deleting
    from it is a no-op rather than a failure. Every other error is real and must
    not be swallowed — a partial delete leaves orphaned nodes behind.
    """
    return "was not found" in str(exc).lower()
