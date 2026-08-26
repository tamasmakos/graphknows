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
from tests.fixtures.memory import FakeMemory as _FakeMemory

_HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


@pytest.mark.asyncio
async def test_recall_reads_last_message_and_returns_memories() -> None:
    mem = _FakeMemory()
    state = {"messages": [{"role": "user", "content": "what is my cat?"}]}
    update = await recall(state, memory=mem, session_id="s1", top_k=3)
    assert update["memories"] == ["Mochi is a ragdoll", "Dr. Nagy is in Budapest"]
    assert update["memories_sources"] == ["stm", "vector"]
    # Measured 27 Jul 2026: the node returned only the two lists above, so an
    # agent could not tell who said a memory or when — LoCoMo temporal accuracy
    # fell 0.808 -> 0.058 on exactly that loss — and the fact-sheet (40 facts on
    # conv-30) was computed and thrown away. Assert on VALUES, not key presence.
    assert [(h.speaker, h.ts) for h in update["memories_hits"]] == [
        ("Gina", "20 January, 2023"),
        ("Jon", "1 February, 2023"),
    ]
    assert update["memories_facts"] == ["Mochi -[IS_A]-> ragdoll"]
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
async def test_recall_says_so_when_a_result_is_not_a_hit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only Memory mints hits, so a non-Hit is broken wiring, not a variation.

    Dropped silently it renders as "nothing recalled" — indistinguishable from
    an empty memory, which is the failure mode this repo keeps paying for.
    """

    class _WrongShape(_FakeMemory):
        async def recall_memory(self, query: str, **kw: object) -> dict:
            return {"hits": [{"text": "a dict, not a Hit"}], "facts": []}

    mem = _WrongShape()
    with caplog.at_level("WARNING", logger="graphknows.integrations.langgraph._hooks"):
        update = await recall(
            {"messages": [{"role": "user", "content": "q"}]}, memory=mem, session_id="s1"
        )

    assert update["memories"] == []
    assert "not Hit objects" in caplog.text or "were not Hit" in caplog.text


@pytest.mark.asyncio
async def test_recall_empty_messages_is_noop() -> None:
    mem = _FakeMemory()
    assert await recall({}, memory=mem, session_id="s1") == {
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

    assert [i.value["text"] for i in items] == ["Mochi is a ragdoll", "Dr. Nagy is in Budapest"]
    assert all(i.created_at is not None and i.updated_at is not None for i in items)
    # The whole hit reaches the store item: a two-field excerpt left a langgraph
    # agent unable to date or attribute anything it read back out.
    assert [(i.value["speaker"], i.value["ts"]) for i in items] == [
        ("Gina", "20 January, 2023"),
        ("Jon", "1 February, 2023"),
    ]


@pytest.mark.skipif(_HAS_LANGGRAPH, reason="langgraph installed — guard not exercised")
def test_store_import_without_langgraph_raises_missing_extra() -> None:
    with pytest.raises(MissingExtraError) as exc:
        from graphknows.integrations.langgraph import GraphKnowsStore  # noqa: F401
    assert exc.value.extra == "langgraph"
