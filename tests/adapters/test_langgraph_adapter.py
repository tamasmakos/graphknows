"""Tests for the LangGraph reference adapter.

The hooks are framework-agnostic (no langgraph dependency) so they're tested
here with a fake memory. GraphKnowsStore requires the ``langgraph`` extra; when
it is absent, importing it must raise MissingExtraError.
"""

from __future__ import annotations

import importlib.util

import pytest

from graphknows.exceptions import MissingExtraError
from graphknows.integrations.langgraph import recall, remember

_HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


class _FakeMemory:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, object]] = []
        self.recall_calls: list[dict] = []

    async def ingest_memory(self, messages: object, *, session_id: str, **kw: object) -> dict:
        self.ingested.append((session_id, messages))
        return {"chunks": 1}

    async def recall_memory(
        self, query: str, *, session_id: str, top_k: int, scope: str, cross_session: bool = False
    ) -> dict:
        self.recall_calls.append(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "scope": scope,
                "cross_session": cross_session,
            }
        )
        return {
            "hits": [
                {"text": "Mochi is a cat", "sources": "vector"},
                {"text": "unrelated", "sources": "stm"},
            ]
        }


@pytest.mark.asyncio
async def test_recall_reads_last_message_and_returns_memories() -> None:
    mem = _FakeMemory()
    state = {"messages": [{"role": "user", "content": "what is my cat?"}]}
    update = await recall(state, memory=mem, session_id="s1", top_k=3)
    assert update == {
        "memories": ["Mochi is a cat", "unrelated"],
        "memories_sources": ["vector", "stm"],
    }
    assert mem.recall_calls[0] == {
        "query": "what is my cat?",
        "session_id": "s1",
        "top_k": 3,
        "scope": "both",
        "cross_session": True,
    }


@pytest.mark.asyncio
async def test_recall_extracts_text_from_multimodal_content() -> None:
    mem = _FakeMemory()
    state = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is my cat?"},
                    {"type": "image_url", "image_url": "x"},
                ],
            }
        ]
    }
    await recall(state, memory=mem, session_id="s1")
    assert mem.recall_calls[0]["query"] == "what is my cat?"


@pytest.mark.asyncio
async def test_recall_empty_messages_is_noop() -> None:
    mem = _FakeMemory()
    assert await recall({}, memory=mem, session_id="s1") == {
        "memories": [],
        "memories_sources": [],
    }
    assert mem.recall_calls == []


@pytest.mark.asyncio
async def test_remember_ingests_last_message() -> None:
    mem = _FakeMemory()
    state = {"messages": [{"role": "user", "content": "My cat is Mochi."}]}
    assert await remember(state, memory=mem, session_id="s1") == {}
    assert mem.ingested == [("s1", [{"role": "user", "content": "My cat is Mochi."}])]


@pytest.mark.skipif(not _HAS_LANGGRAPH, reason="requires the langgraph extra")
@pytest.mark.asyncio
async def test_store_search_returns_constructible_search_items() -> None:
    """GraphKnowsStore.search() must return real SearchItems, not raise.

    created_at/updated_at are required positional fields on langgraph's
    SearchItem; omitting them made every search() call raise TypeError. Nothing
    caught it: langgraph is an optional extra absent from the environment the
    gates run in, so --ignore-missing-imports resolved SearchItem to Any and the
    strict type check passed. This test only bites where the extra is installed
    — the environment where the adapter is actually reachable.
    """
    from langgraph.store.base import SearchOp

    from graphknows.integrations.langgraph import GraphKnowsStore

    store = GraphKnowsStore(_FakeMemory())
    items = await store._search(SearchOp(namespace_prefix=("s1",), query="cat", limit=5))

    assert [i.value["text"] for i in items] == ["Mochi is a cat", "unrelated"]
    assert all(i.created_at is not None and i.updated_at is not None for i in items)


@pytest.mark.skipif(_HAS_LANGGRAPH, reason="langgraph installed — guard not exercised")
def test_store_import_without_langgraph_raises_missing_extra() -> None:
    with pytest.raises(MissingExtraError) as exc:
        from graphknows.integrations.langgraph import GraphKnowsStore  # noqa: F401
    assert exc.value.extra == "langgraph"
