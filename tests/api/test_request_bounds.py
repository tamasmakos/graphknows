"""Request sizes are bounded at the trust boundary, not silently clamped (FR-030).

``Message.content``/document ``text``, each scope id, ``metadata``, ``top_k``
and ``limit`` all reject an over-bound value with
:class:`~graphknows.exceptions.ConfigurationError` — a ``GraphKnowsError``,
per the same "every failure escaping a public entry point" contract
``tests/api/test_public_errors.py`` pins. Every check below runs before any
store or network call, so this stays hermetic.
"""

from __future__ import annotations

import asyncio

import pytest

from graphknows import bounds
from graphknows.exceptions import ConfigurationError
from graphknows.memory import Memory
from graphknows.models.message import Message, Role
from graphknows.retrieval.retriever import DETRetriever

# ---------------------------------------------------------------------------
# The bound checks themselves
# ---------------------------------------------------------------------------


def test_text_length_at_the_bound_is_accepted() -> None:
    bounds.check_text_length("x" * bounds.MAX_TEXT_CHARS, "text")


def test_text_length_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        bounds.check_text_length("x" * (bounds.MAX_TEXT_CHARS + 1), "text")


def test_scope_id_at_the_bound_is_accepted() -> None:
    bounds.check_scope_id("s" * bounds.MAX_SCOPE_ID_CHARS, "session_id")


def test_scope_id_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        bounds.check_scope_id("s" * (bounds.MAX_SCOPE_ID_CHARS + 1), "session_id")


def test_empty_scope_id_is_accepted() -> None:
    """An unset (empty) scope id is the common default, not an oversized one."""
    bounds.check_scope_id("", "session_id")


def test_metadata_under_the_bound_is_accepted() -> None:
    bounds.check_metadata_size({"title": "a small note"})


def test_metadata_none_and_empty_are_accepted() -> None:
    bounds.check_metadata_size(None)
    bounds.check_metadata_size({})


def test_metadata_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        bounds.check_metadata_size({"blob": "x" * bounds.MAX_METADATA_BYTES})


def test_top_k_at_the_bound_is_accepted() -> None:
    bounds.check_top_k(bounds.MAX_TOP_K)


def test_top_k_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        bounds.check_top_k(bounds.MAX_TOP_K + 1)


def test_limit_at_the_bound_is_accepted() -> None:
    bounds.check_limit(bounds.MAX_LIMIT)


def test_limit_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        bounds.check_limit(bounds.MAX_LIMIT + 1)


@pytest.mark.asyncio
async def test_bounded_lets_a_call_finishing_in_time_through() -> None:
    @bounds.bounded
    async def _fast() -> str:
        return "ok"

    assert await _fast() == "ok"


@pytest.mark.asyncio
async def test_bounded_rejects_a_call_over_the_per_call_timeout(monkeypatch) -> None:
    monkeypatch.setattr(bounds, "CALL_TIMEOUT_S", 0.01)

    @bounds.bounded
    async def _hangs() -> None:
        await asyncio.sleep(10)

    with pytest.raises(ConfigurationError):
        await _hangs()


# ---------------------------------------------------------------------------
# Wired into the actual trust boundary — never clamped, rejected before any
# store or network call
# ---------------------------------------------------------------------------


def test_message_content_over_the_bound_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        Message(role=Role.user, content="x" * (bounds.MAX_TEXT_CHARS + 1))


def test_message_content_at_the_bound_is_accepted() -> None:
    Message(role=Role.user, content="x" * bounds.MAX_TEXT_CHARS)


@pytest.mark.asyncio
async def test_ingest_memory_rejects_oversized_document_text() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().ingest_memory(text="x" * (bounds.MAX_TEXT_CHARS + 1))


@pytest.mark.asyncio
async def test_ingest_memory_rejects_oversized_scope_id() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().ingest_memory(text="ok", session_id="s" * (bounds.MAX_SCOPE_ID_CHARS + 1))


@pytest.mark.asyncio
async def test_ingest_memory_rejects_oversized_metadata() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().ingest_memory(text="ok", metadata={"blob": "x" * bounds.MAX_METADATA_BYTES})


@pytest.mark.asyncio
async def test_recall_memory_rejects_top_k_over_the_bound() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().recall_memory("q", top_k=bounds.MAX_TOP_K + 1)


@pytest.mark.asyncio
async def test_recall_memory_rejects_oversized_scope_id() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().recall_memory("q", session_id="s" * (bounds.MAX_SCOPE_ID_CHARS + 1))


@pytest.mark.asyncio
async def test_ltm_entities_rejects_limit_over_the_bound() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().ltm_entities(limit=bounds.MAX_LIMIT + 1)


@pytest.mark.asyncio
async def test_ltm_entities_rejects_oversized_scope_id() -> None:
    with pytest.raises(ConfigurationError):
        await Memory().ltm_entities(session_id="s" * (bounds.MAX_SCOPE_ID_CHARS + 1))


@pytest.mark.asyncio
async def test_retriever_rejects_top_k_over_the_bound() -> None:
    """A direct ``DETRetriever`` caller bypasses the ``Memory`` facade, so the
    retriever carries its own copy of the check (evidence:
    retrieval/retriever.py — top_k multiplies into pool_k and the store's
    k_pre = top_k * 20).
    """
    retriever = DETRetriever(store=None)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        await retriever.retrieve("q", session_id="s1", top_k=bounds.MAX_TOP_K + 1)
