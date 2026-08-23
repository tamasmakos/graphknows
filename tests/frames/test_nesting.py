"""Frame nesting: computing it, persisting it, and reading it back.

A role span that embeds another frame's trigger is the structure a flat role
list cannot hold — "I bought it because my laptop died" is Commerce_buy whose
Explanation is a Causation whose Cause is a Death.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.symbolic.framenet.srl import FrameHit, nested_pairs

# "Tam bought a MacBook because his old laptop died."
#  0         10        20        30        40
BUY = FrameHit("Commerce_buy", "bought", 4, (("Goods", 11, 20), ("Explanation", 21, 49)))
CAUSATION = FrameHit("Causation", "because", 21, (("Cause", 29, 49),))
AGE = FrameHit("Age", "old", 33, (("Entity", 33, 43),))
DEATH = FrameHit("Death", "died", 44, (("Protagonist", 29, 43),))


def test_only_the_outermost_frame_in_a_span_attaches() -> None:
    # Explanation covers Causation, Age and Death. Attaching all three asserts
    # the same nesting at three depths; Causation is the one not sitting inside
    # a sibling's role span.
    pairs = nested_pairs([BUY, CAUSATION, AGE, DEATH])
    assert (0, "Explanation", 1) in pairs
    assert (0, "Explanation", 2) not in pairs
    assert (0, "Explanation", 3) not in pairs


def test_nesting_is_directional_and_skips_self() -> None:
    for parent, _, child in nested_pairs([BUY, CAUSATION, AGE, DEATH]):
        assert parent != child


def test_no_spans_means_no_nesting() -> None:
    assert nested_pairs([FrameHit("Giving", "gave", 0), FrameHit("Death", "died", 9)]) == []


@pytest.mark.asyncio
async def test_write_persists_nesting_between_the_instances_it_wrote() -> None:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[{"nn": "tam"}, {"nn": "laptop"}])
    store = GraphStore(client=client)

    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {"frame": "Commerce_buy", "roles": {"Buyer": ["Tam"], "Goods": ["laptop"]}},
            {"frame": "Death", "roles": {"Protagonist": ["laptop"], "Cause": ["Tam"]}},
        ],
        nesting=[(0, "Explanation", 1)],
    )

    nest_calls = [c for c in client.command.await_args_list if "FRAME_NESTS" in c.args[1]]
    assert len(nest_calls) == 1
    rows = nest_calls[0].kwargs["params"]["rows"]
    assert rows[0]["role"] == "Explanation"
    assert rows[0]["p"].startswith("Commerce_buy:")
    assert rows[0]["c"].startswith("Death:")


@pytest.mark.asyncio
async def test_a_pair_naming_a_dropped_instance_is_dropped_too() -> None:
    # The second instance has no persistable filler (no such entity), so it is
    # never written — the edge to it must not be either.
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[{"nn": "tam"}])
    store = GraphStore(client=client)

    await store.write_frame_instances(
        "s1",
        "c1",
        [
            {"frame": "Commerce_buy", "roles": {"Buyer": ["Tam"], "Goods": ["Tam"]}},
            {"frame": "Death", "roles": {"Protagonist": ["ghost"]}},
        ],
        nesting=[(0, "Explanation", 1)],
    )

    assert not [c for c in client.command.await_args_list if "FRAME_NESTS" in c.args[1]]


@pytest.mark.asyncio
async def test_fact_sheet_surfaces_nesting_as_its_own_line() -> None:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(
        side_effect=[
            [],  # REL
            [{"id": "f1", "frame": "Commerce_buy", "roles": ["Buyer=Tam", "Goods=laptop"]}],
            [{"outer": "Commerce_buy", "role": "Explanation", "inner_frame": "Death"}],
        ]
    )
    store = GraphStore(client=client)

    facts = await store.entity_fact_context(["Tam"])

    assert "Commerce_buy: Buyer=Tam, Goods=laptop" in facts
    assert "Commerce_buy -Explanation-> Death" in facts


@pytest.mark.asyncio
async def test_a_namespace_without_the_nesting_edge_still_gets_its_facts() -> None:
    # FRAME_NESTS exists only where the SRL path ran. On any older namespace the
    # read fails, and the rest of the fact-sheet must survive it.
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(
        side_effect=[
            [],
            [{"id": "f1", "frame": "Giving", "roles": ["Donor=Tam", "Theme=book"]}],
            Exception("type FRAME_NESTS not found"),
        ]
    )
    store = GraphStore(client=client)

    assert await store.entity_fact_context(["Tam"]) == ["Giving: Donor=Tam, Theme=book"]
