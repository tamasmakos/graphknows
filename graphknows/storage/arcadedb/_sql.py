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


def vector_literal(values: Iterable[float]) -> str:
    """Return an embedding as an inline array literal: ``[0.1,0.2]``.

    Inline rather than bound: ArcadeDB rejects a LIST parameter as the value
    of an LSM_VECTOR-indexed property, in SQL and Cypher alike.
    """
    return "[" + ",".join(str(v) for v in values) + "]"


def is_missing_type(exc: Exception) -> bool:
    """True when an error only means the vertex type does not exist yet.

    A database that never ingested a given label has no such type, so deleting
    from it is a no-op rather than a failure. Every other error is real and must
    not be swallowed — a partial delete leaves orphaned nodes behind.
    """
    return "was not found" in str(exc).lower()


# ``forget`` marks a record ``forgotten`` instead of deleting it, so merge logs
# and provenance stay replayable (FR-037). The record is invisible to recall only
# because every read query composes this fragment for each tombstoned alias it
# binds — no store-level seam can splice a predicate into arbitrary query text —
# so this is the single place its text is written.
_FORGOTTEN = "forgotten"


def not_forgotten(alias: str) -> str:
    """Return the tombstone filter for *alias*: ``e.state <> 'forgotten'``."""
    return f"{alias}.state <> {quote(_FORGOTTEN)}"
