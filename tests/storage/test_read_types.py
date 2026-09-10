"""FR-040 — store read methods declare what they touch, the store accumulates it.

The dead-weight check diffs the types a run wrote against the ones it read back,
so "read" has to be an observation of the run: every read method carries a
``read_types`` declaration and updates the store's per-connection set as it
executes. No live ArcadeDB — a stub client is enough to prove the accumulation.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.storage.arcadedb._schema import _CORE_DDL
from graphknows.storage.arcadedb.graph_store import GraphStore

_DECLARED_TYPES = {
    stmt.split()[3]
    for _, stmt in _CORE_DDL
    if stmt.startswith(("CREATE VERTEX TYPE ", "CREATE EDGE TYPE "))
}


def _read_methods() -> dict[str, frozenset[str]]:
    """The read-method registry: every method that declares the types it touches."""
    return {
        name: types
        for name in dir(GraphStore)
        if (types := getattr(getattr(GraphStore, name), "read_types", None)) is not None
    }


class _StubClient:
    """Answers every query with no rows; the store still records what it read."""

    async def query(self, db: str, command: str, **kwargs: Any) -> list[dict]:
        return []


@pytest.fixture
def store() -> GraphStore:
    return GraphStore(_StubClient(), db="test-read-types")  # type: ignore[arg-type]


def test_registry_is_not_empty() -> None:
    assert _read_methods()


@pytest.mark.parametrize(("name", "types"), sorted(_read_methods().items()))
def test_declared_types_exist_in_the_schema(name: str, types: frozenset[str]) -> None:
    assert types, f"{name} declares no types"
    assert types <= _DECLARED_TYPES, f"{name} declares unknown types: {types - _DECLARED_TYPES}"


def test_a_fresh_store_has_read_nothing(store: GraphStore) -> None:
    assert store.read_types == frozenset()


async def test_a_read_accumulates_its_declared_types(store: GraphStore) -> None:
    await store.sources()
    assert store.read_types == frozenset({"SOURCE"})

    await store.get_segments_by_ids(["g1"])
    assert store.read_types == frozenset({"SOURCE", "SEGMENT"})


async def test_read_types_are_per_connection(store: GraphStore) -> None:
    await store.sources()
    other = GraphStore(_StubClient(), db="other")  # type: ignore[arg-type]
    assert other.read_types == frozenset()
