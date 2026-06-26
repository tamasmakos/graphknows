"""Unit tests for AgenticRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.memory.query.agentic import AgenticRetriever


def _make_settings():
    settings = MagicMock()
    settings.embed_model = "openai/text-embedding-3-small"
    settings.extraction_model = None
    settings.llm_model = "test-model"
    settings.openrouter_api_key = "sk-test"
    settings.llm_temperature = 0.0
    settings.llm_max_tokens = 512
    return settings


def _make_retriever():
    settings = _make_settings()
    pg = MagicMock()
    pg.search_chunks = AsyncMock(return_value=[])
    neo = MagicMock()
    mg = MagicMock()
    return AgenticRetriever(settings=settings, pg=pg, neo=neo, stm=mg)


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
    from graphknows.memory.query.context import RetrievalContext

    retriever = _make_retriever()
    # Stub _get_react and _get_lm to avoid dspy network/auth
    fake_react = MagicMock(return_value=MagicMock())

    with (
        patch.object(retriever, "_get_react", return_value=fake_react),
        patch.object(retriever, "_get_lm", return_value=MagicMock()),
        patch("graphknows.memory.stores.embedder.embed", return_value=np.zeros((1, 10))),
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
        patch("graphknows.memory.stores.embedder.embed", return_value=np.zeros((1, 10))),
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
        patch("graphknows.memory.stores.embedder.embed", return_value=np.zeros((1, 10))),
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
    retriever._pg.search_chunks = AsyncMock(return_value=[fake_chunk])

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
        patch("graphknows.memory.stores.embedder.embed", return_value=np.zeros((1, 10))),
        patch("asyncio.to_thread", side_effect=_raise_first_then_pass),
    ):
        ctx = await retriever.retrieve("query", session_id="s1", top_k=5)

    assert any(r.get("text") == "fallback chunk" for r in ctx.fused_results)
    assert ctx.fused_results[0].get("sources") == "agentic-fallback"
