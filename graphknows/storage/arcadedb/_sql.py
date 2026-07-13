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
    """Strip single quotes from *value* so it is safe to embed in a quoted literal."""
    return value.replace("'", "")


def quote(value: str) -> str:
    """Return *value* as a sanitized single-quoted SQL string literal."""
    return f"'{sanitize(value)}'"


def quoted_list(values: Iterable[str]) -> str:
    """Return values as a Cypher/SQL list literal: ``['a','b']``."""
    return "[" + ",".join(quote(v) for v in values) + "]"
