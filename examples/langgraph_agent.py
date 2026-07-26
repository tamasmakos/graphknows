"""Minimal LangGraph agent with graphknows long-term memory.

Run against a live ArcadeDB:

    docker compose up -d arcadedb          # or: make infra-up
    pip install 'graphknows[langgraph,local-embeddings,extraction]'
    python examples/langgraph_agent.py

The graph has two memory nodes wired from ``graphknows.integrations.langgraph``:
``recall`` injects prior memories for the latest user turn, and ``remember``
persists the turn. A real agent would put an LLM call between them.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from graphknows import Memory
from graphknows.integrations.langgraph import recall, remember


class State(TypedDict, total=False):
    """Graph state: the running message list and recalled memory texts."""

    messages: list[dict[str, str]]
    memories: list[str]


def respond(state: State) -> dict[str, Any]:
    """Stand-in for an LLM call: echo the recalled memories."""
    recalled = state.get("memories", [])
    reply = f"(recalled {len(recalled)} memories) " + (recalled[0] if recalled else "nothing yet")
    return {"messages": [*state.get("messages", []), {"role": "assistant", "content": reply}]}


async def main() -> None:
    """Build the graph, run two turns, and print the memory-aware reply."""
    memory = Memory()
    session_id = "langgraph-demo"

    graph = StateGraph(State)
    graph.add_node("recall", partial(recall, memory=memory, session_id=session_id))
    graph.add_node("respond", respond)
    graph.add_node("remember", partial(remember, memory=memory, session_id=session_id))
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_edge("respond", "remember")
    graph.add_edge("remember", END)
    app = graph.compile()

    try:
        await app.ainvoke({"messages": [{"role": "user", "content": "My cat is named Mochi."}]})
        out = await app.ainvoke(
            {"messages": [{"role": "user", "content": "What is my cat's name?"}]}
        )
        print(out["messages"][-1]["content"])
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
