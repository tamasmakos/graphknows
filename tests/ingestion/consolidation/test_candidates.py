"""Candidate generation stays bounded, and says what the bound cut (FR-020)."""

from __future__ import annotations

import pytest

from graphknows.ingestion.consolidation.candidates import (
    MAX_CANDIDATES,
    bounded_candidates,
)

pytestmark = pytest.mark.unit


def test_small_neighbourhood_is_kept_whole() -> None:
    pool = bounded_candidates("e0", ["e1", "e2"])
    assert pool.ids == ("e1", "e2")
    assert pool.truncated == 0


def test_a_large_block_is_capped_and_the_drop_is_counted() -> None:
    neighbours = [f"e{i}" for i in range(1, MAX_CANDIDATES + 8)]
    pool = bounded_candidates("e0", neighbours)
    assert pool.ids == tuple(neighbours[:MAX_CANDIDATES])
    assert pool.truncated == 7


def test_the_entity_never_compares_against_itself() -> None:
    assert bounded_candidates("e0", ["e0", "e1"]).ids == ("e1",)


def test_an_entity_proposed_twice_spends_one_comparison() -> None:
    pool = bounded_candidates("e0", ["e1", "e1", "e2"])
    assert pool.ids == ("e1", "e2")
    assert pool.truncated == 0


def test_proposal_order_is_the_priority_order() -> None:
    assert bounded_candidates("e0", ["blocked", "neighbour"]).ids == ("blocked", "neighbour")
