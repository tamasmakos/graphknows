"""Co-role REL adjacency projected from frame instances.

A flat projection of frame roles was written here once and removed as lossy,
type-minting, and cap-consuming. These pin the three fixes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


def _store(known: list[str]) -> tuple[GraphStore, MagicMock]:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[{"nn": n} for n in known])
    return GraphStore(client=client), client


def _rel_call(client: MagicMock):
    calls = [c for c in client.command.await_args_list if "r:REL" in c.args[1]]
    return calls[0] if calls else None


@pytest.mark.asyncio
async def test_every_ordered_pair_is_linked_not_just_the_agent() -> None:
    store, client = _store(["melanie", "counselor", "hospital"])
    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {
                "frame": "Being_employed",
                "roles": {
                    "Employee": ["Melanie"],
                    "Position": ["counselor"],
                    "Employer": ["hospital"],
                },
            }
        ],
    )
    rows = _rel_call(client).kwargs["params"]["rows"]
    # 3 fillers -> 3 pairs, none dropped. Pair ORDER follows role name
    # (Employee < Employer < Position) and carries no meaning: co-role fillers
    # are siblings of one fact, not a subject and an object.
    assert len(rows) == 3
    assert {frozenset((r["h"], r["t"])) for r in rows} == {
        frozenset(("melanie", "counselor")),
        frozenset(("melanie", "hospital")),
        frozenset(("counselor", "hospital")),
    }


@pytest.mark.asyncio
async def test_the_roles_survive_the_flattening() -> None:
    store, client = _store(["melanie", "counselor"])
    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {
                "frame": "Being_employed",
                "roles": {"Employee": ["Melanie"], "Position": ["counselor"]},
            }
        ],
    )
    row = _rel_call(client).kwargs["params"]["rows"][0]
    assert {row["hr"], row["tr"]} == {"Employee", "Position"}
    assert row["p"] == "Being_employed"


@pytest.mark.asyncio
async def test_one_edge_type_with_the_frame_as_a_property() -> None:
    store, client = _store(["melanie", "counselor"])
    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {
                "frame": "Being_employed",
                "roles": {"Employee": ["Melanie"], "Position": ["counselor"]},
            }
        ],
    )
    cypher = _rel_call(client).args[1]
    # No `Being_employed` edge type minted; predicate is a parameter.
    assert "r:REL {predicate: row.p" in cypher
    assert "Being_employed" not in cypher
    assert "source = 'frame'" in cypher


@pytest.mark.asyncio
async def test_endpoints_are_matched_never_minted() -> None:
    # 'ghost' was never extracted, so no edge may reach it.
    store, client = _store(["melanie"])
    await store.write_frame_instances(
        "s1",
        "c1",
        [{"frame": "Giving", "roles": {"Donor": ["Melanie"], "Theme": ["ghost"]}}],
    )
    assert _rel_call(client) is None  # only one persistable filler -> no pair


@pytest.mark.asyncio
async def test_pairs_are_capped_per_instance() -> None:
    fillers = [f"e{i}" for i in range(7)]
    store, client = _store(fillers)
    await store.write_frame_instances(
        "s1",
        "c1",
        [{"frame": "Wide", "roles": {f"R{i}": [f"e{i}"] for i in range(7)}}],
    )
    assert len(_rel_call(client).kwargs["params"]["rows"]) == 6


@pytest.mark.asyncio
async def test_fact_sheet_excludes_frame_derived_adjacency() -> None:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(side_effect=[[], [], []])
    store = GraphStore(client=client)
    await store.entity_fact_context(["Melanie"])
    rel_query = client.query.await_args_list[0].args[1]
    assert "r.source <> 'frame'" in rel_query
