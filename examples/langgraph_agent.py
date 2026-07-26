"""LangGraph agent with two-tier graphknows memory: STM buffer, then LTM graph.

    docker compose up -d arcadedb
    pip install 'graphknows[langgraph]' langchain-openai
    export GRAPHKNOWS_NAMESPACE=demo OPENROUTER_API_KEY=...
    python examples/langgraph_agent.py --session s1
    python examples/langgraph_agent.py --session s2

Every turn is buffered as a cheap TURN row (the STM buffer) — no extraction,
no embedding. ``mem.flush()`` at the end of a session ingests those buffered
turns through the real pipeline (entities, relations, topics) into the
knowledge graph and marks them consolidated. The next run's ``recall`` searches
that graph namespace-wide, so session ``s2`` can answer using what ``s1`` said.

Session ``s1`` states two facts, then asks about one in the same session —
answered from the still-unflushed STM buffer. Session ``s2`` asks that same
question again (now necessarily from the graph) plus a multi-hop question
needing two facts joined through a shared entity (cat -> vet -> vet's city).

Each turn prints the provenance of every recalled hit (``src=stm``,
``src=vector``, ``src=entity``, ...) so it is visible which channel actually
answered — including when the graph is not the one that did.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from functools import partial
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graphknows.integrations.langgraph import GraphKnowsMemory

TURNS = {
    "s1": [
        "My cat Mochi is a ragdoll.",
        "Mochi's vet is Dr. Nagy, her clinic is in Budapest.",
        "What breed is my cat?",
    ],
    "s2": [
        "What breed is my cat?",
        "Which city is my cat's vet in?",
    ],
}


class State(TypedDict, total=False):
    """Graph state: messages, the last recall's texts, and its provenance."""

    messages: list[dict[str, str]]
    memories: list[str]
    memories_sources: list[str]


async def respond(llm: ChatOpenAI, state: State) -> dict[str, Any]:
    """Answer the latest turn strictly from the recalled memories."""
    context = "\n".join(state.get("memories", [])) or "(nothing recalled)"
    question = state["messages"][-1]["content"]
    prompt = (
        f"Context (recalled memories):\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only from the context above, in one sentence. If it does not "
        "contain the answer, say you don't know."
    )
    reply = await llm.ainvoke(prompt)
    return {"messages": [*state["messages"], {"role": "assistant", "content": str(reply.content)}]}


async def trace(mem: GraphKnowsMemory, state: State) -> dict[str, Any]:
    """Print each recalled hit's channel, then one row of the running census."""
    for text, src in zip(
        state.get("memories", []), state.get("memories_sources", []), strict=False
    ):
        print(f"    src={src or 'none':<10} {text}")
    # stats() scopes TURN/CHUNK to this session but counts ENTITY/REL/TOPIC
    # across the namespace (they are shared, not owned by a session). Printed
    # as two labelled groups so "CHUNK=0 ENTITY=6" does not read as a
    # contradiction — this session has extracted nothing yet, the graph it can
    # already recall from was built by earlier sessions.
    s = await mem.stats()
    print(
        f"    census  this session: TURN(raw)={s['turns']['raw']:<3} "
        f"CHUNK={s['chunks']['consolidated']:<3} | "
        f"namespace: ENTITY={s['entities']:<3} REL={s['relations']:<3} TOPIC={s['topics']}"
    )
    return {}


def build_graph(mem: GraphKnowsMemory, llm: ChatOpenAI) -> Any:
    """Wire the four-node graph: recall -> remember -> respond -> trace.

    ``remember`` runs BEFORE ``respond`` deliberately: both memory hooks read
    the LAST message, so with the reverse order ``remember`` would buffer the
    assistant's own reply and the user's turn would never be stored at all.
    """
    graph = StateGraph(State)
    graph.add_node("recall", mem.recall)
    graph.add_node("remember", mem.remember)
    graph.add_node("respond", partial(respond, llm))
    graph.add_node("trace", partial(trace, mem))
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "remember")
    graph.add_edge("remember", "respond")
    graph.add_edge("respond", "trace")
    graph.add_edge("trace", END)
    return graph.compile()


async def main() -> None:
    """Run one session's scripted turns, then flush it into the graph."""
    parser = argparse.ArgumentParser(description="graphknows + LangGraph memory demo")
    parser.add_argument("--session", required=True, choices=sorted(TURNS))
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("GRAPHKNOWS_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        temperature=0,
    )

    async with GraphKnowsMemory(args.session) as mem:
        app = build_graph(mem, llm)
        for i, turn in enumerate(TURNS[args.session], start=1):
            print(f"\nturn {i}  user: {turn}")
            out = await app.ainvoke({"messages": [{"role": "user", "content": turn}]})
            print(f"    answer: {out['messages'][-1]['content']}")

        if args.flush:
            print(f"\nflush: {await mem.flush()}")


if __name__ == "__main__":
    asyncio.run(main())
