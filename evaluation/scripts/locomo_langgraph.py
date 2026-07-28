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
import re
import time
from datetime import datetime
from typing import Any, TypedDict

from dateparser import parse as _parse_date
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from evaluation.common.llm import extract_answer
from evaluation.common.memories import extract_date
from evaluation.locomo.prompts import ANSWER_PROMPT
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


def parse_turn(message: dict[str, Any], *, prefix: bool = False) -> dict[str, str]:
    """A LoCoMo message -> the three fields graphknows has homes for.

    The speaker and timestamp are METADATA. They are deliberately NOT prepended
    to the text: doing so is what minted them as entities.

    ``prefix=True`` reproduces the pre-metadata baseline EXACTLY — the speaker
    and timestamp are flattened into the text and the metadata fields are left
    empty, so ``CHUNK.speaker`` falls back to the role and ``CHUNK.ts`` stays
    "". That is the arm that scored 0.700, and holding everything else constant
    (same program, same prompt, same renderer, same judge) is what makes the
    clean-vs-prefixed comparison a one-variable test.
    """
    speaker = str(message.get("role", "") or "")
    timestamp = str(message.get("timestamp", "") or "")
    text = str(message.get("content", "") or "").strip()
    if prefix:
        return {"speaker": "", "timestamp": "", "text": f"{timestamp} | {speaker}: {text}"}
    return {"speaker": speaker, "timestamp": timestamp, "text": text}


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
    # A first-person subject IS nameable now that graphknows takes the speaker
    # as a parameter — "I moved to Budapest" is a fact about Gina. This branch
    # used to reject it as "a bare pronoun", which is also why the prefixed arm
    # extracted 98% of turns to this one's 91%: its "Gina:" prefix handed every
    # turn a PROPN subject, so the skip never fired.
    if subject.text.lower() in {"i", "we", "you", "my", "our"}:
        return True, f"first-person clause: {subject.text!r} resolves to the speaker"
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
    memories_hits: list[dict[str, Any]]
    memories_facts: list[str]
    dated: int
    undated: int


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


MAX_CONTEXT_CHARS = 72_000


def render_hit(hit: dict[str, Any]) -> tuple[str, datetime | None]:
    """One hit -> ``("(January 20, 2023) Gina: text", when)``.

    ``ANSWER_PROMPT`` Stage 5 does date arithmetic keyed on a literal
    ``(Month DD, YYYY)`` prefix, and tells the model never to carry a clock time
    into an answer — so the raw ``ts`` ("4:04 pm on 20 January, 2023") is parsed
    down to the day. The speaker is kept because Stage 2 checks attribution; the
    prefixed baseline got that for free with ``Gina:`` inline in the text.

    NO ``ts`` -> the text is passed through untouched. That is the ``--prefix``
    arm, whose text already carries its own date and speaker inline; prefixing
    it again would double the date, and dropping it would render a whole arm
    undated against a prompt that promises dates. Both paths end at the same
    information, which is what makes the two arms comparable.
    """
    text = str(hit.get("text") or "").strip()
    raw_ts = str(hit.get("ts") or "").strip()
    when = _parse_date(raw_ts) if raw_ts else None
    if when is None:
        return text, extract_date(text)
    when = when.replace(tzinfo=None)
    who = str(hit.get("speaker") or "").strip()
    lead = f"({when.strftime('%B %d, %Y')})"
    return (f"{lead} {who}: {text}" if who else f"{lead} {text}"), when


def render(hits: list[dict[str, Any]]) -> tuple[str, int, int]:
    """Hits -> the memory block, plus ``(dated, undated)`` counts.

    Dedup and the character budget run in RELEVANCE order (dropping whole
    memories, never truncating one mid-sentence); only the survivors are then
    sorted chronologically, because the prompt reads the timeline oldest-first
    while the choice of what to keep must follow the ranking. Same contract as
    ``evaluation.common.memories.format_memories`` — reimplemented here rather
    than imported because that one sources its date by regex-scraping the
    passage text, which returns nothing once the date is no longer prefixed
    into it. Every LoCoMo-specific decision stays in this file.

    The counts are returned so a run reports how many memories rendered dated.
    A clean arm silently falling back to the undated branch is the failure this
    whole change exists to prevent; it should be visible, not inferred.
    """
    seen: set[str] = set()
    dated: list[tuple[datetime, str]] = []
    undated: list[str] = []
    used = 0
    for hit in hits:
        block, when = render_hit(hit)
        if not block or block in seen:
            continue
        seen.add(block)
        cost = len(block) + (2 if used else 0)
        if used + cost > MAX_CONTEXT_CHARS:
            continue  # a shorter, lower-ranked memory may still fit
        used += cost
        (dated.append((when, block)) if when else undated.append(block))
    dated.sort(key=lambda pair: pair[0])
    return "\n\n".join([b for _, b in dated] + undated), len(dated), len(undated)


def build_qa_app(mem: GraphKnowsMemory, llm: ChatOpenAI, reference_date: str) -> Any:
    """recall -> answer. One question per invocation.

    The answerer is the harness's staged ``ANSWER_PROMPT`` verbatim. It is built
    on dated, attributed memories and a structured fact-sheet; the three-line
    prompt it replaces threw both away, which is half of why the clean-parse arm
    read as a regression.
    """

    async def answer(state: QAState) -> dict[str, Any]:
        body, n_dated, n_undated = render(state.get("memories_hits", []))
        facts = state.get("memories_facts", [])
        prompt = ANSWER_PROMPT.format(
            reference_date=reference_date or "2023",
            facts=(
                "Known facts (structured, extracted from the conversations):\n"
                + "\n".join(f"- {f}" for f in facts)
                + "\n\n"
                if facts
                else ""
            ),
            memories=body or "(nothing recalled)",
            question=state["messages"][-1]["content"],
        )
        r = await llm.ainvoke(prompt)
        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": extract_answer(str(r.content)) or "Unknown"},
            ],
            "dated": n_dated,
            "undated": n_undated,
        }

    g = StateGraph(QAState)
    g.add_node("recall", mem.recall)
    g.add_node("answer", answer)
    g.add_edge(START, "recall")
    g.add_edge("recall", "answer")
    g.add_edge("answer", END)
    return g.compile()


# ---------------------------------------------------------------- phases ----
async def ingest(
    sessions: list[dict[str, Any]], ns: str, nlp: Any, prefix: bool = False
) -> collections.Counter[str]:
    stats: collections.Counter[str] = collections.Counter()
    for s in sessions:
        sid = s["session_id"]
        t0 = time.monotonic()
        async with GraphKnowsMemory(sid, namespace=ns) as mem:
            app = build_ingest_app(mem, nlp, stats)
            for msg in s["messages"]:
                turn = parse_turn(msg, prefix=prefix)
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


def reference_date_of(sessions: list[dict[str, Any]]) -> str:
    """Newest turn timestamp, human-readable — Stage 5's "around {reference_date}".

    Anchors the prompt's relative-time reasoning to the conversation's own era
    instead of today; the harness derives the same value from session headers.
    """
    seen = [
        parsed.replace(tzinfo=None)
        for s in sessions
        for m in s["messages"]
        if (raw := str(m.get("timestamp") or "").strip()) and (parsed := _parse_date(raw))
    ]
    return max(seen).strftime("%d %B %Y") if seen else ""


def evidence_recall(gold_turns: list[str], memories: list[str]) -> float:
    """Fraction of the gold evidence turns present verbatim in what was recalled.

    THE metric that separates a retrieval miss from a generation miss, and the
    only one that can tell you which half to work on. Without it a wrong answer
    is unattributable: "Unknown" looks identical whether the evidence never
    arrived or arrived and was ignored.

    Same contract as ``evaluation.common.lexical.evidence_recall`` — exact
    substring on whitespace-normalised text, per-hop so a 4-evidence question
    can score 0.25, and it cannot be inflated by retrieving more. Reimplemented
    over hit dicts rather than imported because that one takes RetrievedPassage.

    Scoring only. The gold turns never reach the retriever or the answerer.
    """
    if not gold_turns:
        return -1.0  # no gold evidence — excluded from the mean, never scored 0
    haystack = " ".join(" ".join(m.split()) for m in memories)
    return sum(1 for turn in gold_turns if turn in haystack) / len(gold_turns)


def gold_evidence(q: dict[str, Any], sessions: list[dict[str, Any]]) -> list[str]:
    """``['D1:3']`` -> the utterance CONTENTS. 1-indexed on both axes."""
    ids = [(int(a), int(b)) for a, b in re.findall(r"D(\d+):(\d+)", str(q.get("evidence") or ""))]
    out: list[str] = []
    for doc_no, turn_no in ids:
        try:
            content = sessions[doc_no - 1]["messages"][turn_no - 1].get("content")
        except (IndexError, KeyError, TypeError, AttributeError):
            continue
        if content and (text := " ".join(str(content).split())):
            out.append(text)
    return out


async def answer_questions(
    questions: list[dict[str, Any]],
    ns: str,
    conv: str,
    limit: int,
    reference_date: str,
    sessions: list[dict[str, Any]],
    top_k: int = 25,
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
    async with GraphKnowsMemory(f"{conv}-qa", namespace=ns, top_k=top_k) as mem:
        app = build_qa_app(mem, llm, reference_date)
        for i, q in enumerate(questions[:limit], start=1):
            state = {"messages": [{"role": "user", "content": q["question"]}]}
            ans, score, sources, dated, undated, errored = "", 0.0, [], 0, 0, False
            gold_turns = gold_evidence(q, sessions)
            ev_recall, recalled = -1.0, []
            # One retry: a transient 502 scored as a wrong answer moves the mean
            # by 0.012 per question, which is inside the 0.04 parity threshold
            # these arms are compared at. A real failure still lands as 0.0.
            for attempt in (1, 2):
                try:
                    st = await asyncio.wait_for(app.ainvoke(state), timeout=180)
                    ans = str(st["messages"][-1]["content"])
                    score, _ = await judge.judge_two_stage(q["question"], str(q["answer"]), ans)
                    sources = st.get("memories_sources", [])
                    dated, undated = int(st.get("dated", 0)), int(st.get("undated", 0))
                    recalled = list(st.get("memories", []))
                    ev_recall = evidence_recall(gold_turns, recalled)
                    errored = False
                    break
                except Exception as exc:
                    errored = True
                    print(f"[{i}] {q['id']} ERROR attempt {attempt} {exc!s:.70}", flush=True)
            out.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "gold": str(q["answer"]),
                    "answer": ans,
                    "score": score,
                    "sources": sources,
                    "dated": dated,
                    "undated": undated,
                    "errored": errored,
                    # The triage fields. `evidence_recall` says whether the
                    # evidence ARRIVED; `score` says whether it was USED. A low
                    # score with recall 1.0 is a generation problem, with recall
                    # 0.0 a retrieval problem — and nothing else distinguishes
                    # them. `memories` is kept so a failure can be read directly
                    # instead of re-run.
                    "evidence_recall": ev_recall,
                    "gold_evidence": gold_turns,
                    "memories": recalled,
                }
            )
            print(
                f"[{i:>2}/{min(limit, len(questions))}] {q['id']} "
                f"{q['category']:<20} score={score:.2f}"
                + (f" ev_recall={ev_recall:.2f}" if ev_recall >= 0 else " ev_recall=n/a"),
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
    # Both are arm-validity checks, not decoration. `errors` says how much of
    # this mean is provider failure rather than memory quality; `dated` says
    # whether the memories actually reached the answerer with dates on them —
    # a clean arm falling back to the undated branch would otherwise look like
    # a mysteriously bad score.
    dated, undated = sum(r["dated"] for r in qa), sum(r["undated"] for r in qa)
    shown = dated + undated
    print(f"    errors: {sum(1 for r in qa if r['errored'])}/{len(qa)}")
    print(f"    memories shown: {shown}  dated={dated} ({dated / shown:.0%})  undated={undated}")

    # Retrieval vs generation. Questions with no gold evidence are excluded
    # rather than scored 0 — 4 of LoCoMo's 1540 rows ship none.
    scored = [r for r in qa if r.get("evidence_recall", -1) >= 0]
    if not scored:
        return
    mean_ev = sum(r["evidence_recall"] for r in scored) / len(scored)
    print(f"\n    evidence_recall: {mean_ev:.3f}  (n={len(scored)})")
    got, missed = [r for r in scored if r["evidence_recall"] >= 0.5], []
    missed = [r for r in scored if r["evidence_recall"] < 0.5]
    quadrant = {
        "retrieved + answered": [r for r in got if r["score"] >= 0.5],
        "RETRIEVED, NOT USED  ": [r for r in got if r["score"] < 0.5],
        "not retrieved, lucky ": [r for r in missed if r["score"] >= 0.5],
        "NOT RETRIEVED        ": [r for r in missed if r["score"] < 0.5],
    }
    print("    triage:")
    for label, rows in quadrant.items():
        print(f"      {label} {len(rows):>3}  {len(rows) / len(scored):>5.1%}")
    gen_bound = quadrant["RETRIEVED, NOT USED  "]
    ret_bound = quadrant["NOT RETRIEVED        "]
    print(
        f"    -> of {len(gen_bound) + len(ret_bound)} failures: "
        f"{len(gen_bound)} generation-bound, {len(ret_bound)} retrieval-bound"
    )


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conv", default="conv-30")
    p.add_argument("--namespace", default="lg30")
    p.add_argument("--phase", choices=["ingest", "qa", "both"], default="both")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=25, help="memories recalled per question")
    p.add_argument(
        "--prefix",
        action="store_true",
        help="baseline arm: flatten speaker+timestamp into the text and set no "
        "metadata fields, exactly as the 0.700 run did",
    )
    a = p.parse_args()

    import spacy

    from graphknows.settings import GraphKnowsSettings

    sessions, questions = load(a.conv)
    nlp = spacy.load(GraphKnowsSettings().spacy_model)

    stats: collections.Counter[str] = collections.Counter()
    qa: list[dict[str, Any]] = []
    print(f"arm: {'PREFIXED (baseline replication)' if a.prefix else 'CLEAN (metadata fields)'}")
    if a.phase in ("ingest", "both"):
        stats = await ingest(sessions, a.namespace, nlp, a.prefix)
    if a.phase in ("qa", "both"):
        qa = await answer_questions(
            questions, a.namespace, a.conv, a.limit, reference_date_of(sessions), sessions, a.top_k
        )
    report(stats, qa)


if __name__ == "__main__":
    asyncio.run(main())
