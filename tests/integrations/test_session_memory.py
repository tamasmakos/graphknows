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

from graphknows.integrations.langgraph import GraphKnowsMemory
from tests.fixtures.memory import FakeMemory as _FakeMemory


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
@pytest.mark.parametrize(
    ("incoming", "expected_role", "expected_name"),
    [
        ("user", "user", None),
        ("assistant", "assistant", None),
        ("human", "user", None),  # LangChain HumanMessage.type
        ("ai", "assistant", None),  # LangChain AIMessage.type
        ("Gina", "user", "Gina"),  # a named participant in a group conversation
    ],
)
async def test_remember_maps_foreign_role_labels(
    incoming: str, expected_role: str, expected_name: str | None
) -> None:
    """Role labels from other frameworks must not reach a TURN raw.

    A TURN role is one of user/assistant/system/tool. LangChain's own
    messages report "human"/"ai", and a multi-party transcript uses participant
    names — passing either through raises a pydantic ValidationError on ingest.
    """
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)

    await gm.remember({"messages": [{"role": incoming, "content": "hello"}]})

    _, messages = mem.ingested[0]
    assert messages[0]["role"] == expected_role
    assert messages[0].get("name") == expected_name


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
async def test_recall_returns_a_prompt_ready_context() -> None:
    """The rendered block, not just the parts to render.

    Handing back only hits meant every integration re-assembled the prompt
    context itself, and they disagreed about where a date comes from — the
    ``ts``-dropping bug that cost 0.33 accuracy. The counts come with it so a
    silent fallback to the undated branch is visible rather than mysterious.
    """
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)

    update = await gm.recall({"messages": [{"role": "user", "content": "What breed?"}]})

    assert "Mochi is a ragdoll" in update["memories_context"]
    assert update["memories_dated"] == 2
    assert update["memories_undated"] == 0


@pytest.mark.asyncio
async def test_recall_returns_texts_and_sources() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)
    state = {"messages": [{"role": "user", "content": "What breed is my cat?"}]}

    update = await gm.recall(state)

    assert update["memories"] == ["Mochi is a ragdoll", "Dr. Nagy is in Budapest"]
    assert update["memories_sources"] == ["stm", "vector"]
    # The attribution an agent needs to date and credit a memory reaches state.
    assert [(h.speaker, h.ts) for h in update["memories_hits"]] == [
        ("Gina", "20 January, 2023"),
        ("Jon", "1 February, 2023"),
    ]
    assert update["memories_facts"] == ["Mochi -[IS_A]-> ragdoll"]


@pytest.mark.asyncio
async def test_recall_empty_messages_is_noop() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)

    update = await gm.recall({"messages": []})

    assert update == {
        "memories": [],
        "memories_sources": [],
        "memories_hits": [],
        "memories_facts": [],
        "memories_context": "",
        "memories_dated": 0,
        "memories_undated": 0,
    }
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


@pytest.mark.asyncio
async def test_remember_carries_the_turn_timestamp() -> None:
    """The turn's timestamp is the anchor its relative dates resolve against.

    Without it, "next Friday" in a 2023 conversation resolves against ingestion
    time. Measured on conv-30: 42 of 62 TEMPORAL nodes landed in 2025/2026 for a
    conversation that ran January-July 2023.
    """
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)
    state = {
        "messages": [
            {"role": "user", "content": "See you next Friday", "timestamp": "20 January, 2023"}
        ]
    }

    await gm.remember(state)

    _, messages = mem.ingested[0]
    assert messages[0]["timestamp"] == "20 January, 2023"


@pytest.mark.asyncio
async def test_remember_omits_timestamp_when_absent() -> None:
    mem = _FakeMemory()
    gm = GraphKnowsMemory("s1", memory=mem)

    await gm.remember({"messages": [{"role": "user", "content": "hello"}]})

    _, messages = mem.ingested[0]
    assert "timestamp" not in messages[0]
