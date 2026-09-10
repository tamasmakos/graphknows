"""Tests for DETRetriever: collectors, RRF fusion, and fail-loud error propagation.

The retriever reads one ``GraphStore`` through its golden reads over SEGMENT
and FACT. Collectors are mocked as AsyncMocks on that store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.retrieval.retriever import DETRetriever
from graphknows.settings import get_settings


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    store = MagicMock()
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock, or the (un-swallowed) await error propagates. Defaults
    # return empty; individual tests override the channels they exercise.
    store.sources = AsyncMock(return_value=[{"id": "src-1", "uri": "session:s1"}])
    store.segments_mentioning = AsyncMock(return_value=[])
    store.full_text_search_segments = AsyncMock(return_value=[])
    store.facts_by_predicate = AsyncMock(return_value=[])
    store.search_segments_ann = AsyncMock(return_value=[])
    store.get_segments_by_ids = AsyncMock(return_value={})
    store.neighbor_segments = AsyncMock(return_value={})
    store.resolve_query_entities = AsyncMock(return_value=[])
    store.facts_by_entity = AsyncMock(return_value=[])
    retriever = DETRetriever(store=store)
    return retriever, store


async def test_all_collectors_contribute():
    retriever, store = _make_retriever()

    store.segments_mentioning = AsyncMock(
        return_value=[{"id": "ent-g1", "text": "entity segment", "entities": ["alice"]}]
    )
    store.full_text_search_segments = AsyncMock(
        return_value=[{"id": "bm25-g1", "text": "alice policy text"}]
    )
    store.search_segments_ann = AsyncMock(
        return_value=[{"id": "sem-g1", "text": "semantic segment", "cosine_score": 0.9}]
    )
    store.facts_by_predicate = AsyncMock(
        return_value=[{"id": "f1", "segment_id": "fact-g1", "text": "alice holds a policy"}]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("alice policy", session_id="s1")

    ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert {"ent-g1", "bm25-g1", "sem-g1", "fact-g1"} <= ids
    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["fact-g1"]["sources"] == "fact"


async def test_session_scopes_every_collector_to_the_sessions_sources():
    """A session is a SOURCE: its id resolves through the uri, and every read is scoped."""
    retriever, store = _make_retriever()

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        await retriever.retrieve("alice policy", session_id="s1")

    store.sources.assert_awaited_once_with(uri="session:s1")
    assert store.segments_mentioning.await_args.args[2] == ["src-1"]
    assert store.full_text_search_segments.await_args.args[2] == ["src-1"]
    assert store.search_segments_ann.await_args.args[2] == ["src-1"]
    assert store.facts_by_predicate.await_args.args[2] == ["src-1"]


async def test_no_session_is_namespace_wide():
    retriever, store = _make_retriever()

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        await retriever.retrieve("alice policy", session_id="")

    store.sources.assert_not_awaited()
    assert store.search_segments_ann.await_args.args[2] is None


async def test_fact_context_attached_when_enabled(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "true")
    get_settings.cache_clear()
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=[{"id": "e1", "name": "Jon"}])
    store.facts_by_entity = AsyncMock(
        return_value=[
            {"subject_name": "Jon", "predicate": "p:visiting", "object_name": "Rome"},
            {"subject_name": "Jon", "predicate": "p:visiting", "object_name": "Rome"},
        ]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1")

    assert ctx.facts == ["Jon visiting Rome"]
    store.resolve_query_entities.assert_awaited_once()
    assert store.facts_by_entity.await_args.args[0] == ["e1"]


async def test_fact_context_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "false")
    get_settings.cache_clear()
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=[{"id": "e1", "name": "Jon"}])
    store.facts_by_entity = AsyncMock(
        return_value=[{"subject_name": "Jon", "predicate": "p:visiting", "object_name": "Rome"}]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1")

    assert ctx.facts == []
    store.resolve_query_entities.assert_not_awaited()


async def test_multi_source_segment_ranks_higher():
    retriever, store = _make_retriever()

    store.segments_mentioning = AsyncMock(
        return_value=[{"id": "shared", "text": "shared text", "entities": ["alice"]}]
    )
    store.search_segments_ann = AsyncMock(
        return_value=[
            {"id": "shared", "text": "shared text", "cosine_score": 0.9},
            {"id": "solo", "text": "solo text", "cosine_score": 0.8},
        ]
    )

    # The fused RRF score IS the ranking now — no reranker to neutralise.
    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("alice", session_id="s1")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["shared"]["rrf_score"] > by_id["solo"]["rrf_score"]


async def test_top_k_is_respected():
    retriever, store = _make_retriever()

    store.segments_mentioning = AsyncMock(
        return_value=[{"id": f"e{i}", "text": f"entity {i}"} for i in range(8)]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("query words here", session_id="s1", top_k=3)

    assert len(ctx.fused_results) <= 3


async def test_collector_failure_raises_loudly():
    """A collector failure must propagate — no silent graceful degradation. A
    broken channel should surface immediately, not masquerade as thin recall."""
    retriever, store = _make_retriever()

    store.segments_mentioning = AsyncMock(side_effect=RuntimeError("store down"))

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await retriever.retrieve("query text here", session_id="s1")


async def test_neighbor_expansion_merges_split_message_siblings(monkeypatch):
    """A long ingested message splits across consecutive segments; the answer
    may sit in a sibling of the matched segment, so siblings are stitched INTO
    the matched passage's text (byte order) rather than added as separate
    passages — the passage count is unchanged so the ``[:gen_context_k]`` cut
    keeps the same number of distinct seeds. Off by default (conversational
    context-rot).
    """
    monkeypatch.setattr("graphknows.retrieval.retriever._NEIGHBOR_RADIUS", 1)
    retriever, store = _make_retriever()
    store.search_segments_ann = AsyncMock(
        return_value=[{"id": "g2", "text": "middle of a long message", "cosine_score": 0.9}]
    )
    store.neighbor_segments = AsyncMock(
        return_value={
            "g2": [
                {"id": "g3", "text": "end (answer)", "byte_start": 60},
                {"id": "g1", "text": "start", "byte_start": 0},
            ],
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("q", session_id="s1", top_k=5)

    # Siblings are NOT separate passages — the passage count is unchanged.
    assert [f["chunk_id"] for f in ctx.fused_results] == ["g2"]
    merged = ctx.fused_results[0]
    # Seed text leads, then siblings in byte order — all evidence is present.
    assert merged["text"] == "middle of a long message start end (answer)"
    assert merged["metadata"]["merged_neighbors"] == 2


async def test_neighbor_expansion_dedupes_and_survives_store_failure(monkeypatch):
    monkeypatch.setattr("graphknows.retrieval.retriever._NEIGHBOR_RADIUS", 1)
    retriever, store = _make_retriever()
    store.search_segments_ann = AsyncMock(
        return_value=[{"id": "a", "text": "x", "cosine_score": 0.9}]
    )
    store.neighbor_segments = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("q", session_id="s1", top_k=5)

    # Failure is non-fatal — the base results are still returned.
    assert [f["chunk_id"] for f in ctx.fused_results] == ["a"]


async def test_fused_hits_carry_role_and_observed_at_for_every_channel():
    """Attribution survives fusion — for every channel, not just the bare ones.

    ``_shape_and_rank`` fetches the stored segment for EVERY candidate, so a
    channel that returned only an id still hands the answerer the role and time
    it needs. Asserts on VALUES, not key presence.
    """
    retriever, store = _make_retriever()
    store.search_segments_ann = AsyncMock(
        return_value=[{"id": "sem-g1", "text": "semantic segment", "cosine_score": 0.9}]
    )
    store.facts_by_predicate = AsyncMock(
        return_value=[{"id": "f1", "segment_id": "fact-g1", "text": "on that day"}]
    )
    store.get_segments_by_ids = AsyncMock(
        return_value={
            "sem-g1": {"text": "semantic segment", "role": "Gina", "observed_at": "2023-01-20"},
            "fact-g1": {"text": "on that day", "role": "Jon", "observed_at": "2023-02-01"},
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("When did Gina find that?", session_id="s1")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["sem-g1"]["speaker"] == "Gina"
    assert by_id["sem-g1"]["ts"] == "2023-01-20"
    assert by_id["fact-g1"]["speaker"] == "Jon"
    assert by_id["fact-g1"]["ts"] == "2023-02-01"


async def test_a_hit_carries_the_segments_observed_at():
    """``ts`` must survive from the store row onto the hit.

    The answerer can only date a memory from a field, not from a date that
    happens to be written inside the passage text — which is why turn-fed
    temporal accuracy was 0.077 against 0.962 document-fed before the row's
    own fields were carried through.
    """
    retriever, store = _make_retriever()
    store.search_segments_ann = AsyncMock(
        return_value=[
            {
                "id": "g1",
                "text": "Jon: Lost my job as a banker yesterday.",
                "observed_at": "4:04 pm on 20 January, 2023",
                "cosine_score": 0.9,
            }
        ]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("when did Jon lose his job", session_id="s1")

    hit = next(h for h in ctx.fused_results if h.get("chunk_id") == "g1")
    assert hit["ts"] == "4:04 pm on 20 January, 2023"
