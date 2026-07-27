"""Tests for DETRetriever: collectors, RRF fusion, and fail-loud error propagation.

Golden-layer (Phase 1): the retriever reads a single ``GraphStore`` with a
lifecycle ``state`` filter. Collectors are mocked as AsyncMocks on that store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.retrieval.retriever import DETRetriever


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    store = MagicMock()
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock, or the (un-swallowed) await error propagates. Defaults
    # return empty; individual tests override the channels they exercise.
    store.query_entity_chunks = AsyncMock(return_value=[])
    store.full_text_search_chunks = AsyncMock(return_value=[])
    store.query_temporal_chunks = AsyncMock(return_value=[])
    # The "when did X?" fallback: reached whenever the query states no date,
    # which is every question of that shape.
    store.query_dated_chunks_by_entities = AsyncMock(return_value=[])
    store.search_topics = AsyncMock(return_value=[])
    store.search_topics_with_chunks = AsyncMock(return_value=[])
    store.search_chunks_ann = AsyncMock(return_value=[])
    store.get_chunks_by_ids = AsyncMock(return_value={})
    store.neighbor_chunks = AsyncMock(return_value={})
    store.resolve_query_entities = AsyncMock(return_value=[])
    store.entity_fact_context = AsyncMock(return_value=[])
    retriever = DETRetriever(store=store)
    return retriever, store


@pytest.mark.asyncio
async def test_all_collectors_contribute():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "ent-c1", "text": "entity chunk", "entities": ["alice"]}]
    )
    store.full_text_search_chunks = AsyncMock(
        return_value=[{"chunk_id": "bm25-c1", "text": "alice policy text"}]
    )
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "sem-c1", "text": "semantic chunk", "cosine_score": 0.9}]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("alice policy", session_id="s1", scope="both")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "ent-c1" in chunk_ids
    assert "bm25-c1" in chunk_ids
    assert "sem-c1" in chunk_ids
    assert not any(cid.startswith("topic:") for cid in chunk_ids)


@pytest.mark.asyncio
async def test_fact_context_attached_when_enabled(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "true")
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=["Jon"])
    store.entity_fact_context = AsyncMock(return_value=["Jon visiting Rome"])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1", scope="both")

    assert ctx.facts == ["Jon visiting Rome"]
    store.resolve_query_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_fact_context_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "false")
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=["Jon"])
    store.entity_fact_context = AsyncMock(return_value=["Jon visiting Rome"])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1", scope="both")

    assert ctx.facts == []
    store.resolve_query_entities.assert_not_awaited()


@pytest.mark.asyncio
async def test_multi_source_chunk_ranks_higher():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "shared", "text": "shared text", "entities": ["alice"]}]
    )
    store.search_chunks_ann = AsyncMock(
        return_value=[
            {"chunk_id": "shared", "text": "shared text", "cosine_score": 0.9},
            {"chunk_id": "solo", "text": "solo text", "cosine_score": 0.8},
        ]
    )

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        # Test fusion order deterministically, independent of whether the
        # cross-encoder model happens to be cached in this environment.
        patch("graphknows.retrieval.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("alice", session_id="s1", scope="stm")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["shared"]["rrf_score"] > by_id["solo"]["rrf_score"]


@pytest.mark.asyncio
async def test_no_synthetic_topic_chunks_without_topics():
    retriever, store = _make_retriever()

    store.search_chunks_ann = AsyncMock(
        return_value=[
            {"chunk_id": "plain-hit", "text": "climate policy", "cosine_score": 0.9},
        ]
    )

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        patch("graphknows.retrieval.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("climate policy", session_id="s1", scope="stm")

    assert ctx.fused_results[0]["chunk_id"] == "plain-hit"
    assert not any(fr["chunk_id"].startswith("topic:") for fr in ctx.fused_results)


@pytest.mark.asyncio
async def test_top_k_is_respected():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": f"e{i}", "text": f"entity {i}"} for i in range(8)]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("query words here", session_id="s1", top_k=3)

    assert len(ctx.fused_results) <= 3


@pytest.mark.asyncio
async def test_collector_failure_raises_loudly():
    """A collector failure must propagate — no silent graceful degradation. A
    broken channel should surface immediately, not masquerade as thin recall."""
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(side_effect=RuntimeError("store down"))

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await retriever.retrieve("query text here", session_id="s1")


@pytest.mark.asyncio
async def test_scope_maps_to_state_filter():
    """scope stm/ltm/both must be threaded to the store as the state filter."""
    retriever, store = _make_retriever()

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        await retriever.retrieve("q text", session_id="s1", scope="ltm")

    # semantic_ann calls search_chunks_ann(emb, k, session_id, state).
    _emb, _k, _sid, state = store.search_chunks_ann.await_args.args
    assert state == "consolidated"


@pytest.mark.asyncio
async def test_neighbor_expansion_merges_split_message_siblings(monkeypatch):
    """A long ingested message splits across consecutive chunks; the answer may
    sit in a sibling of the matched chunk, so siblings are stitched INTO the
    matched passage's text (seq order) rather than added as separate passages —
    the passage count is unchanged so the ``[:gen_context_k]`` cut keeps the same
    number of distinct seeds. Off by default (conversational context-rot).
    """
    monkeypatch.setattr("graphknows.retrieval.retriever._NEIGHBOR_RADIUS", 1)
    retriever, store = _make_retriever()
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "c2", "text": "middle of a long message", "cosine_score": 0.9}]
    )
    store.neighbor_chunks = AsyncMock(
        return_value={
            "c2": [
                {"chunk_id": "c1", "text": "start", "seq": 1},
                {"chunk_id": "c3", "text": "end (answer)", "seq": 3},
            ],
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("q", session_id="s1", top_k=5, scope="both")

    # Siblings are NOT separate passages — the passage count is unchanged.
    assert [f["chunk_id"] for f in ctx.fused_results] == ["c2"]
    merged = ctx.fused_results[0]
    # Seed text leads, then siblings in seq order — all evidence is present.
    assert merged["text"] == "middle of a long message start end (answer)"
    assert merged["metadata"]["merged_neighbors"] == 2


@pytest.mark.asyncio
async def test_neighbor_expansion_dedupes_and_survives_store_failure(monkeypatch):
    monkeypatch.setattr("graphknows.retrieval.retriever._NEIGHBOR_RADIUS", 1)
    retriever, store = _make_retriever()
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "a", "text": "x", "cosine_score": 0.9}]
    )
    store.neighbor_chunks = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("q", session_id="s1", top_k=5, scope="both")

    # Failure is non-fatal — the base results are still returned.
    assert [f["chunk_id"] for f in ctx.fused_results] == ["a"]


@pytest.mark.asyncio
async def test_when_question_falls_back_to_dated_chunks_for_its_entities():
    """A "when did X?" query states no date, so the date-keyed path finds nothing.

    Measured on conv-30: the temporal channel produced candidates for 2 of 81
    queries and for 0 of the 26 temporal questions, because
    query_date_candidates() requires a digit-bearing date IN the query — and a
    question asking FOR a date contains none. The fallback keys on the entity
    instead and returns the dated chunks about it.
    """
    retriever, store = _make_retriever()
    store.query_dated_chunks_by_entities = AsyncMock(
        return_value=[{"chunk_id": "c9", "text": "Gina lost her job", "date": "2023-01-19"}]
    )

    result = await retriever.temporal_traversal(
        "When did Gina lose her job?", "s1", None, 5, ["gina", "job"]
    )

    assert list(result) == ["c9"]
    assert result["c9"][1]["sources"] == {"temporal"}
    assert result["c9"][1]["metadata"]["date"] == "2023-01-19"
    # The date-keyed path must NOT have been consulted — there is no date to key on.
    store.query_temporal_chunks.assert_not_called()
    store.query_dated_chunks_by_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_stating_a_date_still_uses_the_date_keyed_path():
    """The original behaviour is untouched when the query does carry a date."""
    retriever, store = _make_retriever()
    store.query_temporal_chunks = AsyncMock(
        return_value=[{"chunk_id": "c1", "text": "on that day", "date": "2023-02-01"}]
    )

    result = await retriever.temporal_traversal(
        "What did Gina find on 1 February, 2023?", "s1", None, 5, ["gina"]
    )

    assert list(result) == ["c1"]
    store.query_temporal_chunks.assert_awaited_once()
    store.query_dated_chunks_by_entities.assert_not_called()


@pytest.mark.asyncio
async def test_fused_hits_carry_speaker_and_ts_even_for_channels_that_set_metadata():
    """Attribution survives fusion — for every channel, not just the bare ones.

    Measured 27 Jul 2026: ``_shape_and_rank`` built a fixed 8-key row that
    dropped ``speaker``/``ts`` outright, and its ``needs_meta`` gate skipped the
    chunk fetch for any candidate whose channel had already set a truthy
    ``metadata`` (temporal sets ``{"date": ...}``, ppr ``{"ppr_mass": ...}``).
    The temporal channel — the one whose hits most need a date — was therefore
    the one guaranteed to lose it. Asserts on VALUES, not key presence.
    """
    retriever, store = _make_retriever()
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "sem-c1", "text": "semantic chunk", "cosine_score": 0.9}]
    )
    # A temporal hit arrives with metadata already set -> the old gate skipped it.
    store.query_dated_chunks_by_entities = AsyncMock(
        return_value=[{"chunk_id": "tmp-c1", "text": "on that day", "date": "2023-02-01"}]
    )
    store.get_chunks_by_ids = AsyncMock(
        return_value={
            "sem-c1": {"text": "semantic chunk", "speaker": "Gina", "ts": "2023-01-20"},
            "tmp-c1": {"text": "on that day", "speaker": "Jon", "ts": "2023-02-01"},
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve(
            "What did Gina find that weekend?", session_id="s1", scope="both"
        )

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["sem-c1"]["speaker"] == "Gina"
    assert by_id["sem-c1"]["ts"] == "2023-01-20"
    assert by_id["tmp-c1"]["speaker"] == "Jon"
    assert by_id["tmp-c1"]["ts"] == "2023-02-01"
    # The metadata the channel set is not clobbered by the fetch.
    assert by_id["tmp-c1"]["metadata"] == {"date": "2023-02-01"}
