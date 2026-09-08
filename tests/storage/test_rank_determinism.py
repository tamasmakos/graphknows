"""FR-032 — the four RRF/BM25 store queries carry ``ORDER BY``.

``query_entity_chunks``, ``full_text_search_chunks``, ``query_temporal_chunks``
and ``query_dated_chunks_by_entities`` back the four baseline collectors in
``retrieval/retriever.py`` (entity, BM25, temporal, "when did X" dated-entity),
each of which turns row POSITION straight into RRF rank via ``enumerate()``.
None of the four queries carries a scoring column, so every match is a tie —
before the fix, which chunks survived the ``LIMIT`` and in what order was
ArcadeDB's storage order, not the query. The fixture below writes more
matching chunks than any query's ``top_k``, with ids written out of
lexicographic order, so a tie-broken re-run has somewhere to disagree with
itself if the ``ORDER BY chunk_id`` clause is ever lost.

Live ArcadeDB only — the point is real row order, which a mock cannot supply.
"""

from __future__ import annotations

import uuid

import pytest

from graphknows.settings import GraphKnowsSettings
from graphknows.storage import build_arcadedb_client
from graphknows.storage.arcadedb.graph_store import GraphStore

_SESSION = "sess-rank-determinism"
_ENTITY = "Melanie"
_TERM = "juniper"
_DATE = "2026-01-01"

# Written out of lexicographic order on purpose: if the ORDER BY clause were
# ever dropped, a query answered straight from insertion/storage order would
# happen to look sorted already and the test would prove nothing.
_WRITE_ORDER = [7, 3, 5, 1, 6, 0, 4, 2]
_TOP_K = 5  # < len(_WRITE_ORDER), so ORDER BY + LIMIT actually decides who is cut


def _chunk_id(prefix: str, i: int) -> str:
    return f"{prefix}-{i:02d}"


async def _write_fixture(store: GraphStore, prefix: str) -> list[str]:
    """Write chunks (all tied on entity, term and date) for every id in ``_WRITE_ORDER``."""
    ids = [_chunk_id(prefix, i) for i in _WRITE_ORDER]
    await store.write_chunks_bulk(
        [
            {
                "chunk_id": cid,
                "session_id": _SESSION,
                "text": f"{_ENTITY} planted a {_TERM} tree.",
                "seq": i,
            }
            for i, cid in zip(_WRITE_ORDER, ids, strict=True)
        ]
    )
    for cid in ids:
        await store.write_entities_bulk(_SESSION, cid, [{"name": _ENTITY}])
        await store.write_temporal(cid, [_DATE])
    return ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rank_queries_are_order_and_set_stable(arcadedb_required: None) -> None:
    """Each of the four store queries returns identical order + result set twice."""
    settings = GraphKnowsSettings()
    client = build_arcadedb_client(settings)
    db = f"mem_test_{uuid.uuid4().hex[:8]}"
    store = GraphStore(client=client, db=db)
    await store.connect()
    try:
        await store.ensure_schema(dims=8)
        await store.ensure_session(_SESSION, started_at="2026-01-01T00:00:00Z")
        all_ids = await _write_fixture(store, "chunk")
        expected_order = sorted(all_ids)[:_TOP_K]

        async def entity_run() -> list[str]:
            rows = await store.query_entity_chunks([_ENTITY], _SESSION, _TOP_K, "raw")
            return [r["chunk_id"] for r in rows]

        async def bm25_run() -> list[str]:
            rows = await store.full_text_search_chunks(
                [_TERM], _SESSION, _TOP_K, require_all=False, state="raw"
            )
            return [r["chunk_id"] for r in rows]

        async def temporal_run() -> list[str]:
            rows = await store.query_temporal_chunks([_DATE], _SESSION, _TOP_K, "raw")
            return [r["chunk_id"] for r in rows]

        async def dated_entity_run() -> list[str]:
            rows = await store.query_dated_chunks_by_entities([_ENTITY], _SESSION, _TOP_K, "raw")
            return [r["chunk_id"] for r in rows]

        for name, run in (
            ("query_entity_chunks", entity_run),
            ("full_text_search_chunks", bm25_run),
            ("query_temporal_chunks", temporal_run),
            ("query_dated_chunks_by_entities", dated_entity_run),
        ):
            first = await run()
            second = await run()
            # Same query twice: identical order and (since the lists are equal)
            # identical result set.
            assert first == second, f"{name}: order changed across runs: {first} != {second}"
            # The ORDER BY is chunk_id ascending, so the LIMIT-truncated result
            # is exactly the lexicographically-smallest top_k ids — not
            # whatever ArcadeDB's storage happened to hand back first.
            assert first == expected_order, f"{name}: not ordered by chunk_id: {first}"
    finally:
        await store.drop_database()
        await store.close()
