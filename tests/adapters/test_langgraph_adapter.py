"""Tests for the LangGraph reference adapter.

The hooks are framework-agnostic (no langgraph dependency) so they're tested
here with a fake memory. GraphKnowsStore requires the ``langgraph`` extra; when
it is absent, importing it must raise MissingExtraError.
"""

from __future__ import annotations

import importlib.util

import pytest

from graphknows.integrations.langgraph import recall, remember
from graphknows.exceptions import MissingExtraError

_HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


class _FakeMemory:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, str]] = []
        self.recall_calls: list[dict] = []

    async def ingest_memory(self, text: str, *, session_id: str, **kw: object) -> dict:
        self.ingested.append((session_id, text))
        return {"chunks": 1}

    async def recall_memory(
        self, query: str, *, session_id: str, top_k: int, scope: str
    ) -> dict:
        self.recall_calls.append(
            {"query": query, "session_id": session_id, "top_k": top_k, "scope": scope}
        )
        return {"hits": [{"text": "Mochi is a cat"}, {"text": "unrelated"}]}


@pytest.mark.asyncio
async def test_recall_reads_last_message_and_returns_memories() -> None:
    mem = _FakeMemory()
    state = {"messages": [{"role": "user", "content": "what is my cat?"}]}
    update = await recall(state, memory=mem, session_id="s1", top_k=3)
    assert update == {"memories": ["Mochi is a cat", "unrelated"]}
    assert mem.recall_calls[0] == {
        "query": "what is my cat?",
        "session_id": "s1",
        "top_k": 3,
        "scope": "both",
    }


@pytest.mark.asyncio
async def test_recall_extracts_text_from_multimodal_content() -> None:
    mem = _FakeMemory()
    state = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "what is my cat?"}, {"type": "image_url", "image_url": "x"}]}
        ]
    }
    await recall(state, memory=mem, session_id="s1")
    assert mem.recall_calls[0]["query"] == "what is my cat?"


@pytest.mark.asyncio
async def test_recall_empty_messages_is_noop() -> None:
    mem = _FakeMemory()
    assert await recall({}, memory=mem, session_id="s1") == {"memories": []}
    assert mem.recall_calls == []


@pytest.mark.asyncio
async def test_remember_ingests_last_message() -> None:
    mem = _FakeMemory()
    state = {"messages": [{"role": "user", "content": "My cat is Mochi."}]}
    assert await remember(state, memory=mem, session_id="s1") == {}
    assert mem.ingested == [("s1", "My cat is Mochi.")]


@pytest.mark.skipif(_HAS_LANGGRAPH, reason="langgraph installed — guard not exercised")
def test_store_import_without_langgraph_raises_missing_extra() -> None:
    with pytest.raises(MissingExtraError) as exc:
        from graphknows.integrations.langgraph import GraphKnowsStore  # noqa: F401
    assert exc.value.extra == "langgraph"
