"""``query_chunks_by_frame_facts`` — the FRAME_INSTANCE -> CHUNK retrieval route."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


def _store(rows_per_call: list[list[dict]]) -> tuple[GraphStore, MagicMock]:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(side_effect=rows_per_call)
    return GraphStore(client=client), client


def _queries(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.query.await_args_list]


@pytest.mark.asyncio
async def test_scores_accumulate_across_the_frame_and_entity_reads() -> None:
    # c1 is reached by BOTH signals (2 + 2), c2 by the entity read alone (3), so
    # c1 outranks c2 despite c2 scoring higher on that one read.
    store, _ = _store(
        [
            [{"chunk_id": "c1", "n": 2}],
            [{"chunk_id": "c1", "n": 2}, {"chunk_id": "c2", "n": 3}],
        ]
    )

    out = await store.query_chunks_by_frame_facts(["Giving"], ["Melanie"], top_k=10)

    assert out == [{"chunk_id": "c1", "score": 4.0}, {"chunk_id": "c2", "score": 3.0}]


@pytest.mark.asyncio
async def test_entity_names_are_matched_on_name_norm() -> None:
    store, client = _store([[], []])
    await store.query_chunks_by_frame_facts([], ["  MELANIE "], top_k=5)
    assert "'melanie'" in _queries(client)[0]


@pytest.mark.asyncio
async def test_each_signal_is_its_own_read() -> None:
    # One OR'd pattern makes ArcadeDB scan FRAME_INSTANCE whole; two narrow reads
    # each use an index.
    store, client = _store([[], []])
    await store.query_chunks_by_frame_facts(["Giving"], ["Melanie"], top_k=5)
    queries = _queries(client)
    assert len(queries) == 2
    assert "fi.frame IN" in queries[0]
    assert "PLAYS_ROLE" in queries[1]


@pytest.mark.asyncio
async def test_empty_inputs_issue_no_reads() -> None:
    store, client = _store([])
    assert await store.query_chunks_by_frame_facts([], [], top_k=5) == []
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_and_state_scope_both_reads() -> None:
    store, client = _store([[], []])
    await store.query_chunks_by_frame_facts(["Giving"], ["Mel"], 5, "s1", "consolidated")
    for q in _queries(client):
        assert "c.session_id = 's1'" in q
        assert "c.state = 'consolidated'" in q


@pytest.mark.asyncio
async def test_top_k_truncates_after_scoring_not_before() -> None:
    store, _ = _store([[{"chunk_id": "low", "n": 1}, {"chunk_id": "high", "n": 9}], []])
    out = await store.query_chunks_by_frame_facts(["Giving"], [], top_k=1)
    assert out == [{"chunk_id": "high", "score": 9.0}]
