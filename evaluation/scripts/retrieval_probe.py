"""Retrieval failure taxonomy probe.

Ingests ONE dialogue, then for every question classifies the retrieval outcome:

  OK             — gold found in top-5 hits  (passing)
  GEN_FAIL       — gold in top-5, generation missed  (not tested here, flagged from gen_replay)
  RANKING        — gold in top-20, not top-5  (scoring/ranking issue)
  VOCAB_MISMATCH — gold-as-query finds a matching chunk, but question-as-query does not
  MISSING        — no query variant finds a chunk containing the gold tokens
                   (ingestion / chunking / embedding failure)

Three probes per question (all use the SAME already-ingested session — no re-ingest):
  1. recall(question, k=20)          → primary retrieval
  2. direct ArcadeDB keyword search  → ground-truth chunk existence
  3. recall(gold_text, k=5)          → vocabulary test

Usage (inside the workspace container):

    python -m evaluation.scripts.retrieval_probe \\
        --dialogue conv-26 \\
        --out evaluation/results/locomo/probe-conv-26.json

Iterative loop:
    edit a retriever/ingestion fix → re-run with --no-ingest (reuse session) → compare taxonomy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.locomo.dataset import build_documents, first_string, read_rows
from graphknows.memory import Memory as GraphKnowsRuntime
from graphknows.models.hit import Hit
from graphknows.settings import GraphKnowsSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "cannot",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "so",
        "to",
        "up",
        "and",
        "but",
        "if",
        "or",
        "nor",
        "yet",
        "either",
        "neither",
    ]
)


def _gold_tokens(gold: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", gold.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _hits_contain_gold(hits: list[Hit], gold_toks: set[str], min_match: int = 1) -> bool:
    if not gold_toks:
        return False
    combined = " ".join(h.text.lower() for h in hits)
    return sum(1 for t in gold_toks if t in combined) >= min_match


# ---------------------------------------------------------------------------
# Direct ArcadeDB existence check
# ---------------------------------------------------------------------------


async def _chunk_exists_in_db(
    runtime: GraphKnowsRuntime,
    session_id: str,
    gold_toks: set[str],
) -> bool:
    """True if ANY chunk for this session contains at least one gold keyword."""
    if not gold_toks:
        return False
    # Use the retriever's execute_cypher for a direct text search.
    retriever = runtime._retriever  # type: ignore[attr-defined]
    if not hasattr(retriever, "execute_cypher"):
        return False  # fallback: can't check
    # ArcadeDB CONTAINS does not support $param — embed token as literal (same
    # pattern as entity traversal). Access neo._qry directly to avoid execute_cypher's
    # {var} rejection while still using $sid as a proper parameter.
    neo = getattr(retriever, "neo", None)
    if neo is None:
        return False
    for tok in sorted(gold_toks, key=len, reverse=True)[:3]:
        safe = tok.replace("'", "")
        try:
            rows = await neo.query(
                f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                f"-[:HAS_CHUNK]->(c:CHUNK) "
                f"WHERE toLower(c.text) CONTAINS '{safe}' "
                f"RETURN c.id AS cid LIMIT 1",
                sid=session_id,
            )
            if rows:
                return True
        except Exception as exc:
            print(f"  probe query failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Per-question classification
# ---------------------------------------------------------------------------


async def _classify(
    runtime: GraphKnowsRuntime,
    session_id: str,
    question: str,
    gold: str,
    gen_result: str | None = None,  # from gen_replay JSON if available
) -> dict[str, Any]:
    gold_toks = _gold_tokens(gold)

    # Probe 1 — standard recall at k=20
    res = await runtime.recall_memory(question, session_id=session_id, top_k=20, scope="ltm")
    hits20 = res.get("hits", [])
    hits5 = hits20[:5]

    gold_in_5 = _hits_contain_gold(hits5, gold_toks)
    gold_in_20 = _hits_contain_gold(hits20, gold_toks)

    # Probe 2 — direct DB existence check
    in_db = await _chunk_exists_in_db(runtime, session_id, gold_toks)

    # Probe 3 — gold-as-query vocabulary test
    gold_query = gold[:120]  # cap to avoid huge embeddings
    gold_res = await runtime.recall_memory(gold_query, session_id=session_id, top_k=5, scope="ltm")
    gold_hits = gold_res.get("hits", [])
    gold_query_finds = _hits_contain_gold(gold_hits, gold_toks)

    # --- Classify ---
    if gold_in_5:
        # Found in top-5 — retrieval succeeded; any failure is generation
        failure = (
            "OK"
            if gen_result is None
            else ("GEN_FAIL" if gen_result.strip().lower() == "unknown" else "OK")
        )
    elif gold_in_20:
        failure = "RANKING"
    elif not in_db and not gold_query_finds:
        failure = "MISSING"
    elif gold_query_finds and not gold_in_20:
        failure = "VOCAB_MISMATCH"
    elif in_db and not gold_query_finds:
        failure = "EMBED_FAIL"
    else:
        failure = "MISSING"

    return {
        "gold_in_5": gold_in_5,
        "gold_in_20": gold_in_20,
        "in_db": in_db,
        "gold_query_finds": gold_query_finds,
        "failure": failure,
        "top3_scores": [round(h.score, 4) for h in hits20[:3]],
        "top3_sources": [h.sources for h in hits20[:3]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _dialogue_id(case_id: str) -> str:
    idx = case_id.rfind("-q")
    return case_id[:idx] if idx > 0 and case_id[idx + 2 :].isdigit() else case_id


async def _probe(args: argparse.Namespace) -> int:
    rows = read_rows(Path(args.data_root) / args.questions)
    dlg = args.dialogue.strip()
    dlg_rows = [
        r
        for r in rows
        if _dialogue_id(first_string(r, ("id", "question_id", "case_id"), "")) == dlg
    ]
    if not dlg_rows:
        print(f"No rows found for dialogue {dlg!r}")
        return 1

    # Load gen_replay results if provided (for GEN_FAIL labelling)
    gen_map: dict[str, str] = {}
    if args.gen_replay:
        for item in json.loads(Path(args.gen_replay).read_text(encoding="utf-8")):
            gen_map[item["case_id"]] = item.get("gen", "")

    settings = GraphKnowsSettings()
    runtime = GraphKnowsRuntime(settings)
    session_id = f"probe-{dlg}"

    if not args.no_ingest:
        documents = build_documents(
            dlg_rows[0], max_documents=args.max_documents, max_chars=args.max_chars
        )
        await runtime.purge_memory(session_id=session_id)
        print(f"[{dlg}] ingesting {len(documents)} docs …", flush=True)
        for idx, doc in enumerate(documents):
            await runtime.ingest_memory(
                text=doc.text,
                session_id=session_id,
                title=doc.title or f"{dlg}-doc-{idx}",
                metadata={"anchor_date": doc.anchor_date},
            )
        print(f"[{dlg}] flushing …", flush=True)
        await runtime.flush()
        print(f"[{dlg}] ready", flush=True)

    out_rows: list[dict[str, Any]] = []
    counts: Counter = Counter()

    for row in dlg_rows:
        case_id = first_string(row, ("id", "question_id", "case_id"), "")
        question = first_string(row, ("question", "query", "prompt"))
        gold = str(row.get("answer") or row.get("gold") or row.get("target") or "")
        category = first_string(row, ("category", "type", "task_type", "question_type"))
        gen_result = gen_map.get(case_id)

        result = await _classify(runtime, session_id, question, gold, gen_result)
        counts[result["failure"]] += 1

        out_rows.append(
            {
                "case_id": case_id,
                "category": category,
                "question": question,
                "gold": gold,
                "gen": gen_result,
                **result,
            }
        )
        print(
            f"  {case_id}  [{result['failure']:15s}]  "
            f"in_db={result['in_db']}  gold_q={result['gold_query_finds']}  "
            f"q={question[:50]!r}",
            flush=True,
        )

    # Summary
    total = len(out_rows)
    print(f"\n=== probe [{dlg}]  n={total} ===")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {label:20s}: {count:4d}  ({pct:.1f}%)")

    # Per-category breakdown
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in out_rows:
        by_cat[r.get("category") or "?"][r["failure"]] += 1
    print("\n  Per question-category:")
    for cat, cnts in sorted(by_cat.items()):
        total_cat = sum(cnts.values())
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(cnts.items(), key=lambda x: -x[1]))
        print(f"    {cat:25s} n={total_cat}  {breakdown}")

    if args.out:
        Path(args.out).write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
        print(f"\nper-case rows -> {args.out}")

    await runtime.close()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dialogue", default="conv-26")
    p.add_argument("--data-root", default="evaluation/data/locomo")
    p.add_argument("--questions", default="questions.jsonl")
    p.add_argument("--max-documents", type=int, default=50)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument(
        "--no-ingest", action="store_true", help="Reuse existing probe session (skip ingest+flush)."
    )
    p.add_argument(
        "--gen-replay", default="", help="Path to gen_replay JSON for GEN_FAIL labelling."
    )
    p.add_argument("--out", default="")
    raise SystemExit(asyncio.run(_probe(p.parse_args())))


if __name__ == "__main__":
    main()
