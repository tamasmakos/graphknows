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
from typing import Any, TypedDict

from dateparser import parse as _parse_date

# langchain/langgraph are imported INSIDE the functions that need them. They
# live in an optional extra the gate environment does not install, and at module
# scope the ImportError aborted pytest COLLECTION — so every test of this file's
# LoCoMo logic silently never ran. Deferring them keeps the schema layer below
# importable, and therefore testable, without the extra.
from evaluation.common.llm import extract_answer
from evaluation.locomo.prompts import ANSWER_PROMPT
from graphknows import Hit, render_memories
from graphknows.channels.base import Channel
from graphknows.channels.temporal import query_date_candidates
from graphknows.integrations.langgraph import GraphKnowsMemory
from graphknows.ranking.rrf import rrf_score

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


# ------------------------------------------------------- custom schema ----
def session_dates(sessions: list[dict[str, Any]]) -> dict[str, str]:
    """``{"conv-30-s1": "2023-01-20", ...}`` — one date per session.

    A LoCoMo session is a single conversation-day: all 28 messages of
    ``conv-30-s1`` carry the identical ``"4:04 pm on 20 January, 2023"``, and 19
    sessions yield exactly 19 distinct timestamps. That is a property of THIS
    dataset, not of conversation memory in general — a support thread can span
    weeks — which is why the session-date layer is built here and not in the
    library.
    """
    out: dict[str, str] = {}
    for s in sessions:
        for m in s["messages"]:
            if (raw := str(m.get("timestamp") or "").strip()) and (when := _parse_date(raw)):
                out[s["session_id"]] = when.strftime("%Y-%m-%d")
                break
    return out


class LoCoMoSessionSchema(Channel):
    """The LoCoMo-shaped graph layer, and the retrieval path it exists for.

    One object holds both halves of the custom schema, which is the point of
    the seam: ``populate`` writes structure during flush, ``collect`` reads it
    during recall. A schema that is written but never read is the failure this
    repo keeps paying for, and keeping both in one class makes that visible.

    What it adds on top of the universal graph:

    ``SESSION-[:ON_DATE]->TEMPORAL``   the conversation-day, as a real node
    ``CHUNK-[:SAID_BY]->ENTITY``       attribution as an edge, not a string
    ``CHUNK.ts`` backfill              every chunk inherits its session's date

    The backfill is the measurable one. 26 of conv-30's 81 questions ask *when*
    something happened and expect a date as the ANSWER, so what decides them is
    not whether retrieval finds the chunk — entity/semantic/BM25 already do —
    but whether the answerer knows that chunk's date. A chunk whose own ``ts``
    came back empty renders undated and the answer is unanswerable from the
    context. Every chunk belongs to a session, and every LoCoMo session has
    exactly one date, so the session is a guaranteed fallback. Watch the
    ``dated=`` percentage in the run report.

    ``SAID_BY`` is deliberately NOT a ``MENTIONS`` edge. Commit c891e7b removed
    the speaker's MENTIONS link to its own chunk because speaker entities appear
    in ~half of all chunks and flooded the entity channel — ``_discriminative_
    entities`` drops anything over 50%. A separate edge type keeps attribution
    traversable without putting Gina and Jon back into that filter's way.
    """

    name = "locomo_session"

    def __init__(self, dates: dict[str, str]) -> None:
        self._dates = dates

    async def populate(self, store: Any, session_id: str) -> None:
        """Write this session's schema layer. Runs once per flush, after chunks exist."""
        iso = self._dates.get(session_id, "")
        # Idempotent DDL, declared by the layer that uses it. ArcadeDB needs an
        # edge type to exist before an edge of it can be created, and these two
        # are ours, not the library's.
        for ddl in (
            "CREATE EDGE TYPE SAID_BY IF NOT EXISTS",
            "CREATE EDGE TYPE ON_DATE IF NOT EXISTS",
        ):
            await store.client.command(store.database, ddl, language="sql")

        if iso:
            # The session's own date, as a first-class TEMPORAL node the same
            # way an in-text date mention is one — so a query can reach a
            # conversation-day without that day being written in any utterance.
            await store.command(
                "MATCH (s:SESSION {session_id: $sid}) "
                "MERGE (d:TEMPORAL {date: $iso}) "
                "MERGE (s)-[:ON_DATE]->(d) "
                "SET s.started_at = $iso",
                sid=session_id,
                iso=iso,
            )
            # Backfill only where the chunk has no date of its own: the turn's
            # own timestamp is more precise when present, and overwriting it
            # would flatten every chunk in a session onto one moment.
            await store.command(
                "MATCH (c:CHUNK {session_id: $sid}) "
                "WHERE c.ts IS NULL OR c.ts = '' "
                "SET c.ts = $iso",
                sid=session_id,
                iso=iso,
            )

        # Attribution: the chunk's speaker string, resolved to the ENTITY that
        # already carries that identity. name_norm is the library's merge key
        # (casefolded, whitespace-collapsed) — matching it is what keeps this
        # pointing at the real person node instead of minting a second one.
        await store.command(
            "MATCH (c:CHUNK {session_id: $sid}) "
            "WHERE c.speaker IS NOT NULL AND c.speaker <> '' "
            "MATCH (e:ENTITY) WHERE e.name_norm = toLower(c.speaker) "
            "MERGE (c)-[:SAID_BY]->(e)",
            sid=session_id,
        )

    async def collect(self, ctx: Any, rt: Any, top_k: int) -> dict[str, Any]:
        """Chunks belonging to a session whose DATE the query names.

        The complement of the built-in temporal channel, which walks
        ``CHUNK-[:MENTIONS_DATE]->TEMPORAL`` and therefore only finds chunks
        that SAY the date. "What did Gina find for her clothing store on 1
        February, 2023?" is answered by a turn that never writes the date down;
        only its session knows it.

        Small by construction: 2 of conv-30's 81 questions name a date. It is
        here because it is the honest read half of this schema, not because it
        is where the accuracy is.
        """
        candidates = query_date_candidates(ctx.query)
        if not candidates:
            return {}
        rows = await rt.store.query(
            "MATCH (d:TEMPORAL)<-[:ON_DATE]-(s:SESSION)<-[:IN_SESSION]-(c:CHUNK) "
            "WHERE d.date IN $dates "
            "RETURN c.chunk_id AS chunk_id LIMIT $limit",
            dates=candidates,
            limit=top_k,
        )
        return {
            str(r["chunk_id"]): (
                rrf_score(rank),
                {"text": "", "entities": [], "sources": {self.name}, "doc_id": "", "metadata": {}},
            )
            for rank, r in enumerate(rows)
            if r.get("chunk_id")
        }


# ------------------------------------------------------------- workflow ----
class IngestState(TypedDict, total=False):
    turn: dict[str, str]
    infer: bool
    reason: str


class QAState(TypedDict, total=False):
    messages: list[dict[str, str]]
    memories: list[str]
    memories_sources: list[str]
    memories_hits: list[Hit]
    memories_facts: list[str]
    dated: int
    undated: int


def build_ingest_app(mem: GraphKnowsMemory, nlp: Any, stats: collections.Counter[str]) -> Any:
    """parse -> classify -> store. One turn per invocation."""
    from langgraph.graph import END, START, StateGraph

    def classify(state: IngestState) -> dict[str, Any]:
        ok, why = should_extract(state["turn"]["text"], nlp)
        stats["extract" if ok else "store_only"] += 1
        return {"infer": ok, "reason": why}

    async def store(state: IngestState) -> dict[str, Any]:
        t = state["turn"]
        await mem.add(
            t["text"],  # clean utterance, no prefix
            speaker=t["speaker"],  # -> CHUNK.speaker
            timestamp=t["timestamp"],  # -> relative-date anchor
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


def build_qa_app(mem: GraphKnowsMemory, llm: Any, reference_date: str) -> Any:
    """recall -> answer. One question per invocation.

    The answerer is the harness's staged ``ANSWER_PROMPT`` verbatim. It is built
    on dated, attributed memories and a structured fact-sheet; the three-line
    prompt it replaces threw both away, which is half of why the clean-parse arm
    read as a regression.

    Rendering those memories is ``graphknows.render_memories``, not something
    this file does: dating, attributing, deduping and budgeting a ranked hit
    list is the library's job on either arm, and the ~60 lines that used to do
    it here disagreed with the harness's own renderer about where a date comes
    from. Both arms still work — a hit with a ``ts`` renders dated and
    attributed, one without (the ``--prefix`` arm) passes through carrying its
    own inline date — because that distinction lives in :class:`Hit`.
    """
    from langgraph.graph import END, START, StateGraph

    async def answer(state: QAState) -> dict[str, Any]:
        memories = render_memories(state.get("memories_hits", []))
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
            memories=memories.text or "(nothing recalled)",
            question=state["messages"][-1]["content"],
        )
        r = await llm.ainvoke(prompt)
        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": extract_answer(str(r.content)) or "Unknown"},
            ],
            "dated": memories.dated,
            "undated": memories.undated,
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
    # The prefixed baseline sets no metadata fields at all, so it must not get
    # the session-date layer either — an arm that dates its chunks from the
    # session while the other dates them from the turn is a two-variable test.
    schema = [] if prefix else [LoCoMoSessionSchema(session_dates(sessions))]
    for s in sessions:
        sid = s["session_id"]
        t0 = time.monotonic()
        async with GraphKnowsMemory(sid, namespace=ns, channels=schema) as mem:
            app = build_ingest_app(mem, nlp, stats)
            for msg in s["messages"]:
                turn = parse_turn(msg, prefix=prefix)
                if turn["text"]:
                    await app.ainvoke({"turn": turn})
            flush = await mem.flush()
            after = await mem.stats()
        cov = flush.get("verifier_coverage") or {}
        scored = sum(t["scored"] for t in cov.values())
        edges = sum(t["edges"] for t in cov.values())
        # Only the tiers the verifier never scored get named: a fully covered
        # run stays one line, an unverifiable tier is impossible to miss.
        unscored = ",".join(k for k, t in cov.items() if t["edges"] and not t["scored"])
        print(
            f"{sid:<14} turns={len(s['messages']):<3} "
            f"extract={stats['extract']:<4} store_only={stats['store_only']:<3} | "
            f"ENTITY={after['entities']:<5} REL={after['relations']:<4} "
            f"TOPIC={after['topics']:<3} {time.monotonic() - t0:.0f}s "
            f"VS={scored}/{edges}{'(!' + unscored + ')' if unscored else ''} "
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
    prefix: bool = False,
) -> list[dict[str, Any]]:
    from langchain_openai import ChatOpenAI

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
    # The same channel object on the read side: its `collect` is the half that
    # reaches a conversation-day no utterance writes down. It needs no dates of
    # its own here — `populate` already put them in the graph. The prefixed
    # baseline never ran `populate`, so it must not carry the channel either:
    # a sixth leg in the fusion is a second variable even when it returns
    # nothing, and this comparison is only worth running at one.
    schema = [] if prefix else [LoCoMoSessionSchema({})]
    async with GraphKnowsMemory(f"{conv}-qa", namespace=ns, top_k=top_k, channels=schema) as mem:
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
    p.add_argument(
        "--namespace",
        default="",
        help="physical database to use; defaults to one per conversation "
        "(conv-30 -> mem_conv_30). Conversations share no entities, so a "
        "shared namespace merges two Jons into one node and lets one "
        "conversation's graph answer another's questions.",
    )
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
    ns = a.namespace or a.conv

    stats: collections.Counter[str] = collections.Counter()
    qa: list[dict[str, Any]] = []
    print(f"arm: {'PREFIXED (baseline replication)' if a.prefix else 'CLEAN (metadata fields)'}")
    print(f"conv={a.conv}  namespace={ns}  sessions={len(sessions)}  questions={len(questions)}")
    if a.phase in ("ingest", "both"):
        stats = await ingest(sessions, ns, nlp, a.prefix)
    if a.phase in ("qa", "both"):
        qa = await answer_questions(
            questions,
            ns,
            a.conv,
            a.limit,
            reference_date_of(sessions),
            sessions,
            a.top_k,
            a.prefix,
        )
    report(stats, qa)


if __name__ == "__main__":
    asyncio.run(main())
