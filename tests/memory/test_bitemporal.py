"""Unit tests for WS-5 bi-temporal substrate.

Tests cover:
1. filter_by_validity_window — pure-logic unit tests (no DB required).
2. Supersedence logic — verify that older single-valued edges are tombstoned
   in :class:`GraphStore.write_relation` (mocked ArcadeDB client). This used to
   guard ``ArcadeDBLTMStore``; the retired two-database split is gone and the
   same supersedence behaviour now lives on the single golden-layer store.

All tests are fast and run without a live ArcadeDB instance.
Integration tests that require a running database are marked
``@pytest.mark.integration`` and skipped by default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.channels.temporal import filter_by_validity_window

# ---------------------------------------------------------------------------
# filter_by_validity_window — window filter logic (pure Python)
# ---------------------------------------------------------------------------


def _edge(
    valid_at: str | None = None,
    invalid_at: str | None = None,
    created_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    e: dict[str, Any] = {}
    if valid_at is not None:
        e["valid_at"] = valid_at
    if invalid_at is not None:
        e["invalid_at"] = invalid_at
    if created_at is not None:
        e["created_at"] = created_at
    e.update(extra)
    return e


class TestFilterByValidityWindow:
    """Pure-logic tests for filter_by_validity_window."""

    def test_open_edge_passes_without_as_of(self) -> None:
        """An edge with no invalid_at (still current) passes with no as_of."""
        edges = [_edge(valid_at="2023-01-01", label="a")]
        out = filter_by_validity_window(edges)
        assert len(out) == 1
        assert out[0]["label"] == "a"
        assert out[0]["_is_current"] is True

    def test_tombstoned_edge_excluded_by_default(self) -> None:
        """An edge with invalid_at in the past is excluded when as_of is not set
        (the reference time is 'now')."""
        edges = [
            _edge(valid_at="2020-01-01", invalid_at="2021-01-01", label="old"),
            _edge(valid_at="2022-01-01", label="current"),
        ]
        # as_of=None → ref_dt = now (2026), so the 2021 tombstone is in the past.
        out = filter_by_validity_window(edges)
        labels = [e["label"] for e in out]
        assert "current" in labels
        assert "old" not in labels

    def test_as_of_window_includes_valid_edge(self) -> None:
        """An edge valid from 2022-01 should be included for as_of=2022-06."""
        edges = [
            _edge(valid_at="2022-01-15", label="job_a"),
            _edge(valid_at="2023-01-01", label="job_b"),
        ]
        out = filter_by_validity_window(edges, as_of="2022-06-01")
        labels = [e["label"] for e in out]
        assert "job_a" in labels
        # job_b hasn't started yet as of 2022-06-01
        assert "job_b" not in labels

    def test_as_of_window_excludes_future_valid_at(self) -> None:
        """Edge with valid_at after the as_of point must be excluded."""
        edges = [_edge(valid_at="2025-01-01", label="future")]
        out = filter_by_validity_window(edges, as_of="2023-01-01")
        assert out == []

    def test_superseded_edge_excluded_before_supersession(self) -> None:
        """Edge tombstoned at 2023-06 should appear for as_of=2022 but not 2023-07."""
        old_job = _edge(valid_at="2020-01-01", invalid_at="2023-06-01", label="old_job")
        new_job = _edge(valid_at="2023-06-01", label="new_job")

        at_2022 = filter_by_validity_window([old_job, new_job], as_of="2022-01-01")
        labels_2022 = [e["label"] for e in at_2022]
        assert "old_job" in labels_2022
        assert "new_job" not in labels_2022

        at_2024 = filter_by_validity_window([old_job, new_job], as_of="2024-01-01")
        labels_2024 = [e["label"] for e in at_2024]
        assert "new_job" in labels_2024
        assert "old_job" not in labels_2024

    def test_temporal_score_attached(self) -> None:
        """_temporal_score must be present on every returned edge."""
        edges = [_edge(valid_at="2020-01-01"), _edge()]
        out = filter_by_validity_window(edges)
        for e in out:
            assert "_temporal_score" in e
            assert 0.0 <= e["_temporal_score"] <= 1.0

    def test_no_valid_at_gets_neutral_score(self) -> None:
        """Edges with no valid_at get the neutral score 0.5."""
        edges = [_edge()]
        out = filter_by_validity_window(edges)
        assert len(out) == 1
        assert out[0]["_temporal_score"] == 0.5

    def test_current_edges_sort_before_tombstoned_for_same_as_of(self) -> None:
        """Current edges rank before tombstoned (but still valid) edges."""
        # Both valid at 2022; only one is tombstoned at 2023.
        tombstoned = _edge(valid_at="2021-01-01", invalid_at="2023-01-01", label="t")
        current = _edge(valid_at="2021-01-01", label="c")
        out = filter_by_validity_window([tombstoned, current], as_of="2022-06-01")
        assert out[0]["label"] == "c"

    def test_valid_from_alias_accepted(self) -> None:
        """``valid_from`` property is accepted as an alias for ``valid_at``."""
        edges = [{"valid_from": "2022-01-01", "label": "alias"}]
        out = filter_by_validity_window(edges)
        assert len(out) == 1
        assert out[0]["label"] == "alias"

    def test_valid_to_alias_accepted(self) -> None:
        """``valid_to`` property is accepted as an alias for ``invalid_at``."""
        edges = [{"valid_from": "2020-01-01", "valid_to": "2021-01-01", "label": "old"}]
        out = filter_by_validity_window(edges, as_of="2024-01-01")
        assert out == []

    def test_empty_input_returns_empty(self) -> None:
        assert filter_by_validity_window([]) == []

    def test_include_open_false_drops_undated_edges(self) -> None:
        """include_open=False with no valid_at drops the edge."""
        edges = [_edge(label="no_date")]
        out = filter_by_validity_window(edges, include_open=False)
        assert out == []


# ---------------------------------------------------------------------------
# Supersedence logic — GraphStore.write_relation (mocked)
# ---------------------------------------------------------------------------


class TestGraphStoreSupersedence:
    """Verify supersedence logic in GraphStore.write_relation via mocked client.

    Formerly ``TestLTMSupersedence`` against the retired ``ArcadeDBLTMStore``.
    The behaviour under test — a new fact for a single-valued predicate
    tombstones the previous current edge — now lives on
    :meth:`GraphStore._supersede_edge` / :meth:`GraphStore.write_relation`.
    """

    def _make_store(self) -> Any:
        """Return a GraphStore with a mocked ArcadeDBClient."""
        from graphknows.storage.arcadedb.graph_store import GraphStore

        client = MagicMock()
        client.command = AsyncMock(return_value=[])
        client.query = AsyncMock(return_value=[])
        client.create_database = AsyncMock(return_value=None)
        store = GraphStore(client=client, db="mem")
        return store, client

    @pytest.mark.asyncio
    async def test_supersedence_called_for_single_valued_predicate(self) -> None:
        """_supersede_edge is called for LIVES_IN before upserting the new edge."""
        store, _client = self._make_store()

        with patch.object(store, "_supersede_edge", new=AsyncMock()) as mock_sup:
            await store.write_relation("Alice", "LIVES_IN", "Berlin", valid_at="2024-01-01")
            mock_sup.assert_called_once()
            args = mock_sup.call_args
            # (predicate, src_norm, dst_norm, now)
            assert args[0][0] == "LIVES_IN"
            assert args[0][1] == "alice"
            assert args[0][2] == "berlin"

    @pytest.mark.asyncio
    async def test_supersedence_not_called_for_multi_valued_predicate(self) -> None:
        """_supersede_edge is NOT called for a multi-valued predicate like KNOWS."""
        store, _client = self._make_store()

        with patch.object(store, "_supersede_edge", new=AsyncMock()) as mock_sup:
            await store.write_relation("Alice", "KNOWS", "Bob")
            mock_sup.assert_not_called()

    @pytest.mark.asyncio
    async def test_temporal_properties_written_on_relation_edge(self) -> None:
        """valid_at is included in the SET clause for relation edges."""
        store, client = self._make_store()

        await store.write_relation("Bob", "WORKS_AT", "Acme", valid_at="2023-03-15")

        # Inspect all command calls and look for one that sets valid_at.
        all_calls = client.command.call_args_list
        rel_edge_calls = [c for c in all_calls if "valid_at" in str(c)]
        assert rel_edge_calls, "Expected at least one command call setting valid_at"

    @pytest.mark.asyncio
    async def test_supersede_edge_issues_correct_cypher(self) -> None:
        """_supersede_edge runs a Cypher SET r.invalid_at on existing edges."""
        store, client = self._make_store()
        client.command = AsyncMock(return_value=[])

        now = "2024-06-01T00:00:00"
        await store._supersede_edge("LIVES_IN", "alice", "paris", now)

        assert client.command.called
        call_args = client.command.call_args
        # Second positional arg is the Cypher string.
        cypher = call_args[0][1]
        assert "invalid_at" in cypher
        assert "is_current" in cypher
