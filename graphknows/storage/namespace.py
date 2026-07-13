"""Namespace → physical ArcadeDB database-name resolution.

A *namespace* is the hard multi-tenant isolation boundary: each namespace maps to
its own pair of ArcadeDB databases (``stm_<ns>`` / ``ltm_<ns>``), so a query in one
namespace physically cannot see another's data. The empty namespace resolves to the
original ``stm`` / ``ltm`` databases, keeping existing single-tenant deployments and
their data unchanged.

This is the single place the naming rule lives — every store, builder, and runtime
derives its database names from :func:`db_names`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

# ArcadeDB database names are filesystem directory names; keep them to a safe
# character set and a conservative length. The suffix we add is ``stm_``/``ltm_``.
_ALLOWED = re.compile(r"[^a-z0-9_]+")
_MAX_NS_LEN = 48


def sanitize(namespace: str) -> str:
    """Normalise a caller-supplied namespace to a safe database-name fragment.

    Lowercases, replaces any character outside ``[a-z0-9_]`` with ``_``, and — when
    the result would be too long for a stable database name — replaces the tail with
    a short hash so distinct inputs stay distinct. The empty/blank namespace maps to
    ``""`` (the default ``stm``/``ltm`` databases).
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


def db_names(namespace: str) -> tuple[str, str]:
    """Return the ``(stm_db, ltm_db)`` database names for a namespace.

    ``""`` → ``("stm", "ltm")`` (backward compatible); ``"acme"`` →
    ``("stm_acme", "ltm_acme")``.
    """
    ns = sanitize(namespace)
    if not ns:
        return ("stm", "ltm")
    return (f"stm_{ns}", f"ltm_{ns}")


def namespaces_from_databases(databases: Iterable[str]) -> list[str]:
    """Derive the available namespaces from a list of ArcadeDB database names.

    Inverse of :func:`db_names` / :func:`db_name`. ``stm_<ns>``/``ltm_<ns>``
    (legacy pair) and ``mem_<ns>`` (single golden-layer database) all map to
    ``<ns>``; a namespace is reported when any of its databases exists. The
    default (``""``) namespace — ``stm``/``ltm``/``mem`` — is always included
    and sorts first, so a browser can always select it.
    """
    found: set[str] = set()
    for db in set(databases):
        for prefix in ("stm_", "ltm_", "mem_"):
            if db.startswith(prefix) and len(db) > len(prefix):
                found.add(db[len(prefix) :])
    return [""] + sorted(found)
