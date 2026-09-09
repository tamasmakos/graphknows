"""Namespace → physical ArcadeDB database-name resolution.

A *namespace* is the hard multi-tenant isolation boundary: each namespace maps to its
own physical ArcadeDB database, so a query in one namespace physically cannot see
another's data.

The golden layer is SINGLE-database: :func:`db_name` maps a namespace to one database
(``mem_<ns>``, or ``mem`` for the default namespace), and that is what every in-package
caller uses — ``memory.py`` and ``storage/__init__.py``. Short-term and long-term memory
are not separate databases: one ingest pipeline writes one graph per namespace.
"""

from __future__ import annotations

import hashlib
import re

# ArcadeDB database names are filesystem directory names; keep them to a safe
# character set and a conservative length. The prefix we add is ``mem_``.
_ALLOWED = re.compile(r"[^a-z0-9_]+")
_MAX_NS_LEN = 48


def sanitize(namespace: str) -> str:
    """Normalise a caller-supplied namespace to a safe database-name fragment.

    Lowercases, replaces any character outside ``[a-z0-9_]`` with ``_``, and — when
    the result would be too long for a stable database name — replaces the tail with
    a short hash so distinct inputs stay distinct. The empty/blank namespace maps to
    ``""`` (the default ``mem`` database).
    """
    ns = (namespace or "").strip().lower()
    if not ns:
        return ""
    ns = _ALLOWED.sub("_", ns).strip("_")
    if not ns:
        return ""
    if len(ns) > _MAX_NS_LEN:
        digest = hashlib.sha256(ns.encode("utf-8")).hexdigest()[:8]
        ns = f"{ns[: _MAX_NS_LEN - 9]}_{digest}"
    return ns


def db_name(namespace: str) -> str:
    """Return the single-database name for a namespace (golden layer).

    ``""`` → ``"mem"``; ``"acme"`` → ``"mem_acme"``.
    """
    ns = sanitize(namespace)
    if not ns:
        return "mem"
    return f"mem_{ns}"
