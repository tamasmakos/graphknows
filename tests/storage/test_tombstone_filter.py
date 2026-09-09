"""Forget is a tombstone, so every read query has to exclude one (FR-037).

``forget`` marks a record ``forgotten`` rather than deleting it, and nothing in
the client can splice a predicate into arbitrary query text — the exclusion is
only as good as the read that composes it. A read query that binds a tombstoned
alias and forgets the filter is the S12 defect this file exists to prevent, so
the tombstoned types are derived from the golden DDL rather than listed here.
"""

from __future__ import annotations

import inspect
import re

from graphknows.storage.arcadedb._schema import _CORE_DDL
from graphknows.storage.arcadedb._sql import not_forgotten
from graphknows.storage.arcadedb.graph_store import GraphStore

# Types whose golden DDL declares the lifecycle ``state`` a tombstone lives in.
TOMBSTONED_TYPES = frozenset(
    m.group(1)
    for _lang, stmt in _CORE_DDL
    if (m := re.search(r"CREATE PROPERTY (\w+)\.state\b", stmt))
)

_READ_METHODS = {name: fn for name, fn in vars(GraphStore).items() if hasattr(fn, "read_types")}


def _bound_aliases(source: str) -> set[str]:
    """Aliases the query text binds to a tombstoned label: ``(e:ENTITY)`` -> ``e``."""
    pattern = rf"\((\w+):({'|'.join(sorted(TOMBSTONED_TYPES))})\b"
    return {m.group(1) for m in re.finditer(pattern, source)}


def _composes_filter(source: str, alias: str) -> bool:
    """True when the query text is built from the shared fragment for *alias*.

    The call, not the rendered SQL: a hand-written ``state <> 'forgotten'`` is
    the copy-paste this fragment exists to replace.
    """
    return any(f"not_forgotten({q}{alias}{q})" in source for q in ("'", '"'))


def test_fragment_is_the_state_comparison() -> None:
    assert not_forgotten("e") == "e.state <> 'forgotten'"


def test_tombstoned_types_come_from_the_golden_schema() -> None:
    assert "ENTITY" in TOMBSTONED_TYPES and "FACT" in TOMBSTONED_TYPES


def test_every_read_query_composes_the_tombstone_filter() -> None:
    missing: dict[str, set[str]] = {}
    for name, method in _READ_METHODS.items():
        source = inspect.getsource(method)
        absent = {a for a in _bound_aliases(source) if not _composes_filter(source, a)}
        if absent:
            missing[name] = absent
    assert not missing, f"read queries missing the tombstone filter: {missing}"
