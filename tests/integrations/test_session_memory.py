"""Tests for GraphKnowsMemory, the session-bound LangGraph convenience wrapper.

The underlying recall/remember hooks are already covered against a fake memory
in tests/adapters/test_langgraph_adapter.py; these tests cover what
GraphKnowsMemory adds on top: binding session settings, and Memory ownership
on __aexit__. No live infra needed — the fake memory implements the same
protocol the hooks depend on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphknows.integrations._session import GraphKnowsMemory


class _FakeMemory:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, object]] = []
        self.recall_calls: list[dict] = []
        self.close = AsyncMock()

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
                {"text": "Mochi is a ragdoll", "sources": "stm"},
                {"text": "Dr. Nagy is in Budapest", "sources": "vector"},
            ]
        }


@pytest.mark.asyncio
async def test_remember_ingests_message_list_not_bare_string() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)
    state = {"messages": [{"role": "user", "content": "My cat Mochi is a ragdoll."}]}

    await gm.remember(state)

    assert len(mem.ingested) == 1
    session_id, messages = mem.ingested[0]
    assert session_id == "s1"
    assert messages == [{"role": "user", "content": "My cat Mochi is a ragdoll."}]


@pytest.mark.asyncio
async def test_remember_preserves_role_from_last_message() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)
    state = {"messages": [{"role": "assistant", "content": "Got it, a ragdoll."}]}

    await gm.remember(state)

    _, messages = mem.ingested[0]
    assert messages[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_recall_is_cross_session_by_default() -> None:
    """A later session must see earlier sessions' graph, or memory is per-session only.

    This is the whole point of flushing to a shared namespace: session s2 asks
    about something only session s1 was told. If cross_session ever silently
    reverts to False, recall confines itself to s2's own subgraph and every
    such question comes back empty — with no error to notice.
    """
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s2", memory=mem)

    await gm.recall({"messages": [{"role": "user", "content": "What breed is my cat?"}]})

    assert mem.recall_calls[0]["cross_session"] is True


@pytest.mark.asyncio
async def test_recall_can_be_confined_to_one_session() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s2", memory=mem, cross_session=False)

    await gm.recall({"messages": [{"role": "user", "content": "What breed is my cat?"}]})

    assert mem.recall_calls[0]["cross_session"] is False


@pytest.mark.asyncio
async def test_recall_returns_texts_and_sources() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)
    state = {"messages": [{"role": "user", "content": "What breed is my cat?"}]}

    update = await gm.recall(state)

    assert update == {
        "memories": ["Mochi is a ragdoll", "Dr. Nagy is in Budapest"],
        "memories_sources": ["stm", "vector"],
    }


@pytest.mark.asyncio
async def test_recall_empty_messages_is_noop() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)

    update = await gm.recall({"messages": []})

    assert update == {"memories": [], "memories_sources": []}
    assert mem.recall_calls == []


@pytest.mark.asyncio
async def test_aexit_closes_self_built_memory_but_not_caller_supplied() -> None:
    # Caller-supplied Memory: __aexit__ must not close it.
    supplied = _FakeMemory()
    gm_supplied = GraphKnowsMemory("s1", memory=supplied)
    async with gm_supplied:
        pass
    supplied.close.assert_not_awaited()

    # Self-built Memory: __aexit__ must close it.
    gm_owned = GraphKnowsMemory("s2")
    gm_owned._memory.close = AsyncMock()  # type: ignore[method-assign]
    async with gm_owned:
        pass
    gm_owned._memory.close.assert_awaited_once()
