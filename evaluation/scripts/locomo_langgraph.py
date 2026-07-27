"""LoCoMo through the LangGraph memory integration, as explicit workflow states.

    docker compose up -d arcadedb
    export OPENROUTER_API_KEY=...
    python -m evaluation.scripts.locomo_langgraph --conv conv-30 --namespace lg30

Everything LoCoMo-specific lives HERE. ``graphknows`` is handed clean text plus
metadata and asked for hits back; it is never told anything about this dataset.

The workflow is two compiled graphs over one namespace:

    INGEST  (per turn)      parse -> classify -> store -> trace
    SESSION END             flush  (buffered turns -> knowledge graph)
    RETRIEVE (per question) recall -> answer -> judge

``parse`` is the step that was missing. A LoCoMo turn arrives already
structured — speaker, timestamp, utterance — and flattening it into
``"4:04 pm on 20 January, 2023 | Gina: Hey Jon!"`` made the extractor re-derive
the speaker and the date AS ENTITIES. Measured on conv-30: "Jon" never existed
as a node, but "Hey Jon", "Thanks, Jon!" and "Oh no, Jon!" did, across 15
fragments. Each field now goes where it belongs: text stays text, the speaker
becomes ``Message.name`` (-> ``CHUNK.speaker``), the timestamp becomes
``Message.timestamp`` (-> the anchor relative dates resolve against).

``classify`` decides whether a turn is worth extracting from. graphknows has no
opinion about that — it exposes ``infer`` and the caller decides. "Hey Jon!" is
stored and embedded and stays retrievable as text; it just mints no entities.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import time
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graphknows.integrations.langgraph import GraphKnowsMemory

DATA = "evaluation/data/locomo/questions.jsonl"


# ---------------------------------------------------------------- parse ----
def load(conv: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (sessions, questions) for one LoCoMo conversation."""
    with open(DATA, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    qs = [r for r in rows if r.get("id", "").startswith(f"{conv}-q")]
    if not qs:
        raise SystemExit(f"no questions for {conv} in {DATA}")
    return qs[0]["sessions"], qs


def parse_turn(message: dict[str, Any]) -> dict[str, str]:
    """A LoCoMo message -> the three fields graphknows has homes for.

    The speaker and timestamp are METADATA. They are deliberately NOT prepended
    to the text: doing so is what minted them as entities.
    """
    return {
        "speaker": str(message.get("role", "") or ""),
        "timestamp": str(message.get("timestamp", "") or ""),
        "text": str(message.get("content", "") or "").strip(),
    }


# ------------------------------------------------------------- classify ----
def should_extract(text: str, nlp: Any) -> tuple[bool, str]:
    """Is this turn worth extracting from? ``(yes, reason)``.

    A turn earns extraction when it predicates something about something.
    Naming a person does not count — "Hey Jon!" and "Mochi's vet is Dr. Nagy"
    BOTH contain a PERSON entity, and only the second asserts anything; the
    vocative has no clause at all. Skipped turns are still stored and embedded,
    so a wrong skip costs graph coverage, never the evidence.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False, "empty"
    doc = nlp(stripped)

    subject = next((t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), None)
    if subject is None:
        # Speech drops subjects constantly ("Still working on opening a dance
        # studio"). An asserting verb plus a content noun is an elliptical clause.
        verb = next((t for t in doc if t.pos_ == "VERB"), None)
        noun = next((t for t in doc if t.pos_ == "NOUN"), None)
        if verb is not None and (noun is not None or doc.ents):
            about = noun.text if noun is not None else doc.ents[0].text
            return True, f"elliptical clause: {verb.lemma_!r} + {about!r}"
        return False, "no clause with a subject (greeting, vocative, backchannel)"

    if subject.pos_ in {"NOUN", "PROPN"}:
        return True, f"clause about {subject.text!r}"
    if (noun := next((t for t in doc if t.pos_ == "NOUN"), None)) is not None:
        return True, f"clause mentioning {noun.text!r}"
    return False, "clause subject is a bare pronoun with no content noun"


# ------------------------------------------------------------- workflow ----
class IngestState(TypedDict, total=False):
    turn: dict[str, str]
    infer: bool
    reason: str


class QAState(TypedDict, total=False):
    messages: list[dict[str, str]]
    memories: list[str]
    memories_sources: list[str]


def build_ingest_app(mem: GraphKnowsMemory, nlp: Any, stats: collections.Counter[str]) -> Any:
    """parse -> classify -> store. One turn per invocation."""

    def classify(state: IngestState) -> dict[str, Any]:
        ok, why = should_extract(state["turn"]["text"], nlp)
        stats["extract" if ok else "store_only"] += 1
        return {"infer": ok, "reason": why}

    async def store(state: IngestState) -> dict[str, Any]:
        t = state["turn"]
        await mem.memory.add(
            [
                {
                    "role": "user",
                    "name": t["speaker"],  # -> CHUNK.speaker
                    "content": t["text"],  # clean utterance, no prefix
                    "timestamp": t["timestamp"],  # -> relative-date anchor
                }
            ],
            run_id=mem.session_id,
            infer=state["infer"],
        )
        return {}

    g = StateGraph(IngestState)
    g.add_node("classify", classify)
    g.add_node("store", store)
    g.add_edge(START, "classify")
    g.add_edge("classify", "store")
    g.add_edge("store", END)
    return g.compile()


def build_qa_app(mem: GraphKnowsMemory, llm: ChatOpenAI) -> Any:
    """recall -> answer. One question per invocation."""

    async def answer(state: QAState) -> dict[str, Any]:
        ctx = "\n".join(state.get("memories", [])) or "(nothing recalled)"
        q = state["messages"][-1]["content"]
        r = await llm.ainvoke(
            f"Memories:\n{ctx}\n\nQuestion: {q}\n\nAnswer from the memories above, "
            "as briefly as possible — a name, a date, or one short phrase. If they "
            "do not contain the answer, reply exactly: Unknown"
        )
        return {"messages": [*state["messages"], {"role": "assistant", "content": str(r.content)}]}

    g = StateGraph(QAState)
    g.add_node("recall", mem.recall)
    g.add_node("answer", answer)
    g.add_edge(START, "recall")
    g.add_edge("recall", "answer")
    g.add_edge("answer", END)
    return g.compile()


# ---------------------------------------------------------------- phases ----
async def ingest(sessions: list[dict[str, Any]], ns: str, nlp: Any) -> collections.Counter[str]:
    stats: collections.Counter[str] = collections.Counter()
    for s in sessions:
        sid = s["session_id"]
        t0 = time.monotonic()
        async with GraphKnowsMemory(sid, namespace=ns) as mem:
            app = build_ingest_app(mem, nlp, stats)
            for msg in s["messages"]:
                turn = parse_turn(msg)
                if turn["text"]:
                    await app.ainvoke({"turn": turn})
            flush = await mem.flush()
            after = await mem.stats()
        print(
            f"{sid:<14} turns={len(s['messages']):<3} "
            f"extract={stats['extract']:<4} store_only={stats['store_only']:<3} | "
            f"ENTITY={after['entities']:<5} REL={after['relations']:<4} "
            f"TOPIC={after['topics']:<3} {time.monotonic() - t0:.0f}s "
            f"err={len(flush['errors'])}",
            flush=True,
        )
    return stats


async def answer_questions(
    questions: list[dict[str, Any]], ns: str, conv: str, limit: int
) -> list[dict[str, Any]]:
    from evaluation.common.dspy_judge import DSPyJudge

    judge = DSPyJudge(os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ.get("GRAPHKNOWS_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        temperature=0,
        timeout=120,
        max_retries=2,
    )
    out: list[dict[str, Any]] = []
    async with GraphKnowsMemory(f"{conv}-qa", namespace=ns, top_k=25) as mem:
        app = build_qa_app(mem, llm)
        for i, q in enumerate(questions[:limit], start=1):
            try:
                st = await asyncio.wait_for(
                    app.ainvoke({"messages": [{"role": "user", "content": q["question"]}]}),
                    timeout=180,
                )
                ans = str(st["messages"][-1]["content"])
                score, _ = await judge.judge_two_stage(q["question"], str(q["answer"]), ans)
                sources = st.get("memories_sources", [])
            except Exception as exc:  # provider hiccup: one datapoint, not the run
                print(f"[{i}] {q['id']} ERROR {exc!s:.80}", flush=True)
                ans, score, sources = "", 0.0, []
            out.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "gold": str(q["answer"]),
                    "answer": ans,
                    "score": score,
                    "sources": sources,
                }
            )
            print(
                f"[{i:>2}/{min(limit, len(questions))}] {q['id']} "
                f"{q['category']:<20} score={score:.2f}",
                flush=True,
            )
            with open(f"/tmp/locomo_lg_{conv}.json", "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
    return out


def report(stats: collections.Counter[str], qa: list[dict[str, Any]]) -> None:
    total = stats["extract"] + stats["store_only"]
    if total:
        print(
            f"\ningest: {total} turns  extracted={stats['extract']} "
            f"({stats['extract'] / total:.0%})  store-only={stats['store_only']}"
        )
    if not qa:
        return
    print(f"\nQA: n={len(qa)}  accuracy={sum(r['score'] for r in qa) / len(qa):.3f}")
    by = collections.defaultdict(list)
    for r in qa:
        by[r["category"]].append(r["score"])
    for c, xs in sorted(by.items()):
        print(f"    {c:<22} n={len(xs):<3} {sum(xs) / len(xs):.3f}")
    ch: collections.Counter[str] = collections.Counter()
    for r in qa:
        for s in r["sources"]:
            for p in str(s).split(","):
                if p:
                    ch[p] += 1
    print(f"    channels: {dict(ch.most_common())}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conv", default="conv-30")
    p.add_argument("--namespace", default="lg30")
    p.add_argument("--phase", choices=["ingest", "qa", "both"], default="both")
    p.add_argument("--limit", type=int, default=1000)
    a = p.parse_args()

    import spacy

    from graphknows.settings import GraphKnowsSettings

    sessions, questions = load(a.conv)
    nlp = spacy.load(GraphKnowsSettings().spacy_model)

    stats: collections.Counter[str] = collections.Counter()
    qa: list[dict[str, Any]] = []
    if a.phase in ("ingest", "both"):
        stats = await ingest(sessions, a.namespace, nlp)
    if a.phase in ("qa", "both"):
        qa = await answer_questions(questions, a.namespace, a.conv, a.limit)
    report(stats, qa)


if __name__ == "__main__":
    asyncio.run(main())
