"""Unit tests for WS-6 additions to ArcadeDBLTMStore.

Covers:
- persist_centrality: PageRank / degree_centrality written to ENTITY nodes.
- compute_and_persist_pagerank: power-iteration result persisted (mocked graph).
- prune_tombstoned_edges: only tombstoned-beyond-window edges deleted.
- prune_duplicate_edges: only low-confidence tombstoned edges deleted.

All tests use a mocked ArcadeDBClient — no live DB required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest


# ---------------------------------------------------------------------------
# Fixture: mocked store
# ---------------------------------------------------------------------------

def _make_store() -> tuple[Any, MagicMock]:
    """Return (ArcadeDBLTMStore, mock_client)."""
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore

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


# ---------------------------------------------------------------------------
# Flush pipeline integration: new phases wired correctly
# ---------------------------------------------------------------------------

class TestFlushPipelineWS6Phases:
    """Verify flush.py calls the WS-6 phases with a fully mocked store."""

    def _make_flush(self, extraction_uses_llm: bool = False) -> Any:
        from graphknows.memory.flush import LTMPromotionPipeline

        stm_store = MagicMock()
        stm_store.connect = AsyncMock(return_value=None)
        stm_store.close = AsyncMock(return_value=None)
        stm_store.read_session_subgraph = AsyncMock(
            return_value={
                "entities": [],
                "files": [],
                "chunks": [],
                "dialogs": [],
                "turns": [],
                "relation_edges": [
                    {"source": "Alice", "relation": "WORKS_AT", "target": "Acme", "confidence": 0.9}
                ],
                "chunk_entity_edges": [],
                "turn_entity_edges": [],
                "next_edges": [],
            }
        )
        stm_store.delete_session = AsyncMock(return_value=None)

        vec_store = MagicMock()
        vec_store.connect = AsyncMock(return_value=None)
        vec_store.close = AsyncMock(return_value=None)
        vec_store.promote_to_ltm = AsyncMock(return_value=(0, 0))
        vec_store.upsert_entities = AsyncMock(return_value=None)
        vec_store.clear_stm = AsyncMock(return_value=None)

        ltm_store = MagicMock()
        ltm_store.connect = AsyncMock(return_value=None)
        ltm_store.close = AsyncMock(return_value=None)
        ltm_store.write_session_subgraph = AsyncMock(return_value={"nodes": 2, "edges": 1})
        ltm_store.compute_and_persist_pagerank = AsyncMock(return_value=2)
        ltm_store.prune_tombstoned_edges = AsyncMock(return_value=0)
        ltm_store.prune_duplicate_edges = AsyncMock(return_value=0)

        pipeline = LTMPromotionPipeline(
            stm_store=stm_store,
            vec_store=vec_store,
            ltm_store=ltm_store,
            llm_api_key="" if not extraction_uses_llm else "sk-test",
            llm_model="deepseek/deepseek-v4-flash",
            extraction_uses_llm=extraction_uses_llm,
        )
        return pipeline, stm_store, vec_store, ltm_store

    @pytest.mark.asyncio
    async def test_pagerank_persistence_called(self) -> None:
        from graphknows.memory.models.promotion import PromotionRequest

        pipeline, _, _, ltm_store = self._make_flush(extraction_uses_llm=False)
        req = PromotionRequest(session_id="s1", run_gds=False, clear_stm=False)
        await pipeline.promote(req)
        ltm_store.compute_and_persist_pagerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_pruning_called(self) -> None:
        from graphknows.memory.models.promotion import PromotionRequest

        pipeline, _, _, ltm_store = self._make_flush(extraction_uses_llm=False)
        req = PromotionRequest(session_id="s1", run_gds=False, clear_stm=False)
        await pipeline.promote(req)
        ltm_store.prune_tombstoned_edges.assert_called_once()
        ltm_store.prune_duplicate_edges.assert_called_once()

    @pytest.mark.asyncio
    async def test_phases_non_fatal_when_store_lacks_methods(self) -> None:
        """If ltm_store doesn't expose new methods, pipeline must not crash."""
        from graphknows.memory.flush import LTMPromotionPipeline
        from graphknows.memory.models.promotion import PromotionRequest

        stm_store = MagicMock()
        stm_store.connect = AsyncMock(return_value=None)
        stm_store.close = AsyncMock(return_value=None)
        stm_store.read_session_subgraph = AsyncMock(
            return_value={
                "entities": [], "files": [], "chunks": [], "dialogs": [], "turns": [],
                "relation_edges": [], "chunk_entity_edges": [], "turn_entity_edges": [], "next_edges": [],
            }
        )
        stm_store.delete_session = AsyncMock(return_value=None)

        vec_store = MagicMock()
        vec_store.connect = AsyncMock(return_value=None)
        vec_store.close = AsyncMock(return_value=None)
        vec_store.promote_to_ltm = AsyncMock(return_value=(0, 0))
        vec_store.upsert_entities = AsyncMock(return_value=None)
        vec_store.clear_stm = AsyncMock(return_value=None)

        # Minimal ltm_store without any WS-6 methods (but with the store lifecycle
        # methods the promotion pipeline now requires of every store).
        ltm_store = MagicMock(spec=[
            "write_session_subgraph",
            "connect",
            "close",
        ])
        ltm_store.connect = AsyncMock(return_value=None)
        ltm_store.close = AsyncMock(return_value=None)
        ltm_store.write_session_subgraph = AsyncMock(return_value={"nodes": 0, "edges": 0})
        # Ensure WS-6 methods are absent
        del ltm_store.compute_and_persist_pagerank
        del ltm_store.prune_tombstoned_edges
        del ltm_store.prune_duplicate_edges

        pipeline = LTMPromotionPipeline(
            stm_store=stm_store,
            vec_store=vec_store,
            ltm_store=ltm_store,
            llm_api_key="",
            llm_model="",
            extraction_uses_llm=False,
        )
        req = PromotionRequest(session_id="s1", run_gds=False, clear_stm=False)
        # Must not raise
        record = await pipeline.promote(req)
        assert record.session_id == "s1"
