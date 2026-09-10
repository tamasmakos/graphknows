"""Replay and undo from the ``MERGED_INTO`` log alone (SC-007, FR-017)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from evaluation.scenarios import SCENARIOS
from graphknows.ingestion.consolidation.entity_resolution import (
    EntityView,
    Layer,
    Link,
    Resolver,
    replay,
    undo,
)

pytestmark = pytest.mark.unit

AT = datetime(2026, 9, 10, tzinfo=UTC)


def _log(*pairs: tuple[str, str]) -> tuple[Link, ...]:
    """A ``MERGED_INTO`` log of *pairs*, one decision per minute in order."""
    return tuple(
        Link(
            source_id=source,
            target_id=target,
            layer=Layer.STRONG,
            evidence=f"key={target}",
            score=1.0,
            at=AT + timedelta(minutes=minute),
        )
        for minute, (source, target) in enumerate(pairs)
    )


def _views(*ids: str) -> dict[str, EntityView]:
    return {
        entity_id: EntityView(id=entity_id, name_norm="acme", type_histogram={"org": rank + 1})
        for rank, entity_id in enumerate(ids)
    }


def test_replay_reconstructs_a_single_merge() -> None:
    assert replay(_log(("acme2", "acme"))) == {"acme2": "acme"}


def test_replay_follows_a_chain_to_its_survivor() -> None:
    """A merged id whose target was itself merged resolves to the end of the chain."""
    assert replay(_log(("a", "b"), ("b", "c"))) == {"a": "c", "b": "c"}


def test_replay_reads_the_log_in_decision_order_not_argument_order() -> None:
    """The log is the record; the order it is handed back in must not matter."""
    log = _log(("a", "b"), ("b", "c"))
    assert replay(reversed(log)) == replay(log)


def test_undo_reverses_one_merge_and_leaves_the_rest() -> None:
    log = _log(("a", "b"), ("b", "c"))
    assert undo(log, log[1]) == {"a": "b"}


def test_a_planned_merge_replays_from_its_log_without_the_views() -> None:
    """The plan's log alone reproduces the merge it decided (SC-007)."""
    views = _views("acme1", "acme2")
    plan = Resolver().plan(views, {"acme1": ["acme2"]})
    assert replay(plan.merges) == {"acme1": "acme2"}


def test_merge_replay_scenario_passes() -> None:
    """The ``merge-replay`` cutover scenario holds (FR-042)."""
    result = asyncio.run(SCENARIOS["merge-replay"]("scenario"))
    assert result.passed, result.detail
