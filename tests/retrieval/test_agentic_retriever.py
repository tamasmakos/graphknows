"""Unit tests for AgenticRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.settings import MemoryMode
from graphknows.retrieval.agentic import AgenticRetriever


def _make_settings():
    settings = MagicMock()
    settings.embed_model = "BAAI/bge-small-en-v1.5"
    settings.extraction_model = None
    settings.llm_model = "test-model"
    settings.llm_api_key = "sk-test"
    settings.llm_api_base = ""
    settings.llm_temperature = 0.0
    settings.llm_max_tokens = 512
    settings.mode = MemoryMode.llm_assisted
    settings.enable_frames = False
    return settings


def _make_retriever():
    settings = _make_settings()
    store = MagicMock()
    store.search_chunks_ann = AsyncMock(return_value=[])
    store._qry = AsyncMock(return_value=[])
    return AgenticRetriever(settings=settings, store=store)


async def _fake_to_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _stub_retriever_internals(retriever, fake_react=None):
    """Patch _get_react and _get_lm so tests skip all dspy network setup."""
    react = fake_react or MagicMock(return_value=MagicMock())
    lm = MagicMock()
    retriever._react = react
    retriever._lm = lm
    retriever._last_session_id = None  # allow session-based rebuild tests to work
    return react, lm


@pytest.mark.asyncio
async def test_retrieve_returns_retrieval_context():
    from graphknows.retrieval.models.context import RetrievalContext

    retriever = _make_retriever()
    # Stub _get_react and _get_lm to avoid dspy network/auth
    fake_react = MagicMock(return_value=MagicMock())

    with (
        patch.object(retriever, "_get_react", return_value=fake_react),
        patch.object(retriever, "_get_lm", return_value=MagicMock()),
        patch("graphknows.storage.embedder.embed", return_value=np.zeros((1, 10))),
        patch("asyncio.to_thread", side_effect=_fake_to_thread),
    ):
        ctx = await retriever.retrieve("test query", session_id="s1", top_k=5)

    assert isinstance(ctx, RetrievalContext)
    assert ctx.query == "test query"
    assert ctx.session_id == "s1"


@pytest.mark.asyncio
async def test_lm_not_rebuilt_on_same_session_call():
    """LM instance must be reused across calls within the same session."""
    retriever = _make_retriever()
    fake_lm = MagicMock()
    call_count = 0

    def _get_lm_counted():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            retriever._lm = fake_lm
        return retriever._lm

    with (
        patch.object(retriever, "_get_lm", side_effect=_get_lm_counted),
        patch.object(retriever, "_get_react", return_value=MagicMock(return_value=MagicMock())),
        patch("graphknows.storage.embedder.embed", return_value=np.zeros((1, 10))),
        patch("asyncio.to_thread", side_effect=_fake_to_thread),
    ):
        await retriever.retrieve("query one", session_id="s1", top_k=3)
        await retriever.retrieve("query two", session_id="s1", top_k=3)

    # _get_lm called once per retrieve() call but the underlying LM object is the same
    assert retriever._lm is fake_lm


@pytest.mark.asyncio
async def test_react_not_rebuilt_within_same_session():
    """_get_react should return same ReAct module for same session."""
    retriever = _make_retriever()
    get_react_calls = []

    def _get_react_counting(*, session_id=""):
        get_react_calls.append(session_id)
        return MagicMock(return_value=MagicMock())

    with (
        patch.object(retriever, "_get_react", side_effect=_get_react_counting),
        patch.object(retriever, "_get_lm", return_value=MagicMock()),
        patch("graphknows.storage.embedder.embed", return_value=np.zeros((1, 10))),
        patch("asyncio.to_thread", side_effect=_fake_to_thread),
    ):
        await retriever.retrieve("q1", session_id="s1", top_k=3)
        await retriever.retrieve("q2", session_id="s1", top_k=3)
        await retriever.retrieve("q3", session_id="s2", top_k=3)

    # Called once per retrieve() — sessions s1, s1, s2
    assert get_react_calls == ["s1", "s1", "s2"]


@pytest.mark.asyncio
async def test_react_failure_falls_back_to_vector_search():
    """If DSPy ReAct raises, retrieve() still returns chunks from vector fallback."""
    retriever = _make_retriever()
    fake_chunk = {
        "chunk_id": "c1",
        "text": "fallback chunk",
        "doc_id": "d1",
        "cosine_score": 0.8,
        "metadata": {},
    }
    retriever._store.search_chunks_ann = AsyncMock(return_value=[fake_chunk])

    call_count = 0

    async def _raise_first_then_pass(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("DSPy error")
        return fn(*args, **kwargs)

    with (
        patch.object(retriever, "_get_react", return_value=MagicMock(return_value=MagicMock())),
        patch.object(retriever, "_get_lm", return_value=MagicMock()),
        patch("graphknows.storage.embedder.embed", return_value=np.zeros((1, 10))),
        patch("asyncio.to_thread", side_effect=_raise_first_then_pass),
    ):
        ctx = await retriever.retrieve("query", session_id="s1", top_k=5)

    assert any(r.get("text") == "fallback chunk" for r in ctx.fused_results)
    assert ctx.fused_results[0].get("sources") == "agentic-fallback"


@pytest.mark.asyncio
async def test_agentic_hit_scores_are_discriminative_not_flat():
    """RC-1 regression guard: fused agentic hit-scores must preserve the
    deterministic cosine-fused relevance magnitude (vary across passages), not
    collapse to a flat positional-RRF curve that starves the generator of any
    relevance signal. Pure positional RRF still varies trivially by rank, so we
    assert the score SPREAD tracks the input relevance (≫ positional-only spread).
    """
    from graphknows.retrieval.models.context import RetrievalContext

    retriever = _make_retriever()
    hop_chunks = [
        {"chunk_id": f"c{i}", "doc_id": "d", "text": f"t{i}", "metadata": {}, "rrf_score": s}
        for i, s in enumerate([0.85, 0.42, 0.13, 0.04])
    ]

    def _fake_react(**kwargs):
        retriever._collected_chunks.extend(hop_chunks)
        return MagicMock()

    # base DET pass adds nothing extra; community-weight lookup is a no-op.
    det = MagicMock()
    det.retrieve = AsyncMock(
        return_value=RetrievalContext(
            query="q", session_id="s1", fused_results=[], search_type="DET"
        )
    )
    retriever._det = det
    retriever._store._qry = AsyncMock(return_value=[])

    with (
        patch.object(retriever, "_get_react", return_value=MagicMock(side_effect=_fake_react)),
        patch.object(retriever, "_get_lm", return_value=MagicMock()),
        patch("graphknows.retrieval.query.decompose.decompose", return_value=["q"]),
        patch("asyncio.to_thread", side_effect=_fake_to_thread),
    ):
        ctx = await retriever.retrieve("q", session_id="s1", top_k=4)

    scores = [r["rrf_score"] for r in ctx.fused_results]
    assert len(scores) >= 2, f"expected multiple passages, got {scores}"
    # Fix preserves DET relevance (top≈0.85, spread≈0.81); the old positional-only
    # bug produced top≈1/61≈0.016 with spread≈0.0008.
    assert max(scores) > 0.1, f"top hit-score collapsed to positional RRF: {scores}"
    assert (max(scores) - min(scores)) > 0.1, f"hit-scores flat/non-discriminative: {scores}"
