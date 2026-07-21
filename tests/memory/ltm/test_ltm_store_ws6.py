"""Unit tests for WS-6 additions to ArcadeDBLTMStore.

Covers:
- persist_centrality: PageRank / degree_centrality written to ENTITY nodes.
- prune_tombstoned_edges: only tombstoned-beyond-window edges deleted.
- prune_duplicate_edges: only low-confidence tombstoned edges deleted.

All tests use a mocked ArcadeDBClient — no live DB required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixture: mocked store
# ---------------------------------------------------------------------------

def _make_store() -> tuple[Any, MagicMock]:
    """Return (ArcadeDBLTMStore, mock_client)."""
    from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore

    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    client.create_database = AsyncMock(return_value=None)
    store = ArcadeDBLTMStore(client=client)
    store._declared_edge_types = set()
    return store, client
class TestPersistCentrality:
    @pytest.mark.asyncio
    async def test_empty_input_returns_zero(self) -> None:
        store, _ = _make_store()
        result = await store.persist_centrality({})
        assert result == 0

    @pytest.mark.asyncio
    async def test_writes_pagerank_to_entity(self) -> None:
        store, client = _make_store()
        centrality = {
            "Alice": {"pagerank": 0.42, "degree_centrality": 0.3},
            "Bob": {"pagerank": 0.15},
        }
        result = await store.persist_centrality(centrality)
        assert result == 2
        # Check that at least one command set pagerank
        all_calls = [str(c) for c in client.command.call_args_list]
        assert any("pagerank" in c for c in all_calls)

    @pytest.mark.asyncio
    async def test_skips_entity_with_empty_name(self) -> None:
        store, client = _make_store()
        centrality = {"": {"pagerank": 0.5}}
        result = await store.persist_centrality(centrality)
        assert result == 0
        client.command.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_present_metrics_written(self) -> None:
        """Only keys present in the scores dict are included in SET clause."""
        store, client = _make_store()
        centrality = {"Alice": {"pagerank": 0.3}}
        await store.persist_centrality(centrality)
        all_calls = [str(c) for c in client.command.call_args_list]
        # pagerank SET should be in calls
        assert any("pagerank" in c for c in all_calls)
        # betweenness NOT in calls (not provided)
        assert not any("betweenness" in c for c in all_calls)

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self) -> None:
        """A failure on one entity doesn't block the rest."""
        store, client = _make_store()
        call_count = 0

        async def _command(db: str, cypher: str, params: dict | None = None) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ArcadeDB transient error")
            return []

        client.command = _command

        centrality = {
            "Alice": {"pagerank": 0.3},
            "Bob": {"pagerank": 0.2},
        }
        # Should not raise; Alice fails, Bob succeeds
        result = await store.persist_centrality(centrality)
        assert result == 1  # Only Bob written


# ---------------------------------------------------------------------------
# prune_tombstoned_edges
# ---------------------------------------------------------------------------

class TestPruneTombstonedEdges:
    @pytest.mark.asyncio
    async def test_prune_issues_delete_with_cutoff(self) -> None:
        store, client = _make_store()
        # Return a deleted count
        client.command = AsyncMock(return_value=[{"deleted": 5}])
        result = await store.prune_tombstoned_edges(retention_days=90)
        assert result == 5
        # Verify the query contains is_current and invalid_at
        call_args = client.command.call_args
        cypher = call_args[0][1]
        assert "is_current" in cypher
        assert "invalid_at" in cypher

    @pytest.mark.asyncio
    async def test_prune_only_touches_tombstoned_edges(self) -> None:
        """The DELETE WHERE clause must require is_current = false."""
        store, client = _make_store()
        client.command = AsyncMock(return_value=[{"deleted": 0}])
        await store.prune_tombstoned_edges(retention_days=30)
        call_args = client.command.call_args
        cypher = call_args[0][1]
        # is_current = false must be a filter (not just mentioned)
        assert "is_current = false" in cypher

    @pytest.mark.asyncio
    async def test_prune_returns_zero_on_db_error(self) -> None:
        store, client = _make_store()
        client.command = AsyncMock(side_effect=RuntimeError("DB error"))
        # Must be non-fatal
        result = await store.prune_tombstoned_edges()
        assert result == 0

    @pytest.mark.asyncio
    async def test_prune_returns_zero_when_none_returned(self) -> None:
        store, client = _make_store()
        client.command = AsyncMock(return_value=[])
        result = await store.prune_tombstoned_edges()
        assert result == 0


# ---------------------------------------------------------------------------
# prune_duplicate_edges
# ---------------------------------------------------------------------------

class TestPruneDuplicateEdges:
    @pytest.mark.asyncio
    async def test_prune_duplicates_deletes_low_confidence_tombstoned(self) -> None:
        store, client = _make_store()
        client.command = AsyncMock(return_value=[{"deleted": 3}])
        result = await store.prune_duplicate_edges()
        assert result == 3
        call_args = client.command.call_args
        cypher = call_args[0][1]
        assert "is_current = false" in cypher
        # Should check weight or confidence thresholds
        assert "weight" in cypher or "confidence" in cypher

    @pytest.mark.asyncio
    async def test_prune_duplicates_non_fatal_on_error(self) -> None:
        store, client = _make_store()
        client.command = AsyncMock(side_effect=RuntimeError("fail"))
        result = await store.prune_duplicate_edges()
        assert result == 0
