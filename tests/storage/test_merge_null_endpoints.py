"""``merge_entities`` must not re-point a relation onto a null-named endpoint.

The merge replays every edge of the loser onto the winner through
``_repoint_relation``. The other endpoint comes out of a query row as
``row["dst"] or row["dst_nn"]`` — and a row where *both* are NULL used to be
passed straight through as ``None``. ``_repoint_relation`` then wrote the edge
against an entity with no name, i.e. the entity-hygiene repair manufactured
exactly the junk node it exists to remove, silently, at flush time.

This is a unit test on purpose: the defect is in the row-filtering branch, not
in ArcadeDB, so it must fail without live infrastructure.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore

pytestmark = pytest.mark.unit


class _FakeStore(GraphStore):
    """GraphStore with the database replaced by canned rows."""

    def __init__(self, out_rows: list[dict[str, Any]], in_rows: list[dict[str, Any]]) -> None:
        # Deliberately skip GraphStore.__init__: this exercises one branch of
        # merge_entities and needs no client, db or connection.
        self._out_rows = out_rows
        self._in_rows = in_rows
        self.repointed: list[tuple[str, str]] = []
        self.commands: list[str] = []

    async def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN e.name AS name" in cypher:
            # Both the loser and the winner lookup — same shape, distinct names.
            nn = params.get("nn", "")
            return [{"name": nn.title(), "aliases": []}]
        if "-[r:REL]->(b:ENTITY)" in cypher:
            return self._out_rows
        if "(a:ENTITY)-[r:REL]->" in cypher:
            return self._in_rows
        return []

    async def command(self, cypher: str, **params: Any) -> Any:
        self.commands.append(cypher)
        return None

    async def _repoint_relation(self, src_name: str, dst_name: str, row: dict[str, Any]) -> bool:
        self.repointed.append((src_name, dst_name))
        # Mirrors the real signature: True == repointed with no provenance by
        # either route, which merge_entities counts into its log line.
        return not row.get("evidence") and not row.get("frame_instance_id")


def _rel(**over: Any) -> dict[str, Any]:
    row = {
        "dst": None,
        "dst_nn": None,
        "src": None,
        "src_nn": None,
        "predicate": "HAS",
        "fact": "",
        "evidence": "",
        "confidence": 0.5,
        "valid_at": None,
        "source": "svo",
        "verifier_score": None,
    }
    row.update(over)
    return row


async def test_outgoing_edge_with_no_destination_name_is_dropped() -> None:
    store = _FakeStore(out_rows=[_rel()], in_rows=[])

    assert await store.merge_entities("winner", "loser") is True
    assert store.repointed == [], (
        "an edge whose destination has neither name nor name_norm was re-pointed; "
        "the endpoint would be written as None"
    )


async def test_incoming_edge_with_no_source_name_is_dropped() -> None:
    store = _FakeStore(out_rows=[], in_rows=[_rel()])

    assert await store.merge_entities("winner", "loser") is True
    assert store.repointed == []


async def test_named_endpoints_are_still_repointed_in_both_directions() -> None:
    """The guard must not swallow the edges the merge exists to carry over."""
    store = _FakeStore(
        out_rows=[_rel(dst="Clothing Store", dst_nn="clothing store")],
        in_rows=[_rel(src="Jon", src_nn="jon")],
    )

    assert await store.merge_entities("winner", "loser") is True
    assert store.repointed == [("Winner", "Clothing Store"), ("Jon", "Winner")]


async def test_name_norm_alone_is_enough_to_repoint() -> None:
    """Only the display name is missing — name_norm is a usable endpoint."""
    store = _FakeStore(out_rows=[_rel(dst=None, dst_nn="clothing store")], in_rows=[])

    assert await store.merge_entities("winner", "loser") is True
    assert store.repointed == [("Winner", "clothing store")]
