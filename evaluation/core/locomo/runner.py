"""LoCoMo benchmark runner — per-dialogue session batching + category-aware scoring.

Two scoring modes
-----------------
retrieval-only (default, ``generate_answers=False``)
    Score the retrieved context directly against the gold answer using
    category-specific heuristics (exact-match / token-recall / unscored).
    Fast, no LLM calls at scoring time, but NOT paper-comparable.

generation (``generate_answers=True``)
    retrieve → DSPy generate answer → symmetric token F1 vs gold.
    Matches the official LoCoMo paper methodology (arXiv 2402.17753).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from evaluation.core.client import GraphknowsMCPClient
from evaluation.core.file_backed import (
    FileBackedBenchmarkOptions,
    _case_documents,
    _first_string,
    _parse_memory_query_hits,
    _read_rows,
)
from evaluation.core.models import BenchmarkCaseResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring — retrieval-only mode (category-specific heuristics)
# ---------------------------------------------------------------------------

_SCORE_EXACT = "exact_match"
_SCORE_TOKEN_RECALL = "token_recall"
_SCORE_UNSCORED = "unscored"
_SCORE_TOKEN_F1 = "token_f1"

_CATEGORY_STRATEGY: dict[str, str] = {
    "single_hop": _SCORE_EXACT,
    "temporal": _SCORE_EXACT,
    "knowledge_synthesis": _SCORE_TOKEN_RECALL,
    "open_ended": _SCORE_UNSCORED,
}


def _answer_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _token_recall(gold: str, haystack: str) -> float:
    """Token-level recall: fraction of gold answer tokens found in the haystack."""
    gold_toks = _answer_tokens(gold)
    hay_toks = _answer_tokens(haystack)
    if not gold_toks:
        return 0.0
    return len(gold_toks & hay_toks) / len(gold_toks)


def _exact_match_score(gold: str, haystack: str) -> float:
    """Substring + token-overlap scoring (existing behaviour for single_hop / temporal)."""
    expected = gold.strip().lower()
    if not expected:
        return 0.0
    hay = haystack.lower()
    if expected in hay:
        return 1.0
    exp_toks = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", expected))
    if not exp_toks:
        return 0.0
    hay_toks = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", hay))
    return len(exp_toks & hay_toks) / len(exp_toks)


def _score_case(category: str, gold: str, recall: list[dict[str, Any]]) -> tuple[float | None, str]:
    """Retrieval-only scoring (used when generate_answers=False)."""
    strategy = _CATEGORY_STRATEGY.get(category, _SCORE_EXACT)
    if strategy == _SCORE_UNSCORED:
        return None, _SCORE_UNSCORED
    haystack = "\n".join(str(h.get("text", "")) for h in recall)
    if strategy == _SCORE_TOKEN_RECALL:
        return _token_recall(str(gold), haystack), _SCORE_TOKEN_RECALL
    return _exact_match_score(str(gold), haystack), _SCORE_EXACT


# ---------------------------------------------------------------------------
# Scoring — generation mode (symmetric token F1, paper-comparable)
# ---------------------------------------------------------------------------


def _token_f1(gold: str, predicted: str) -> float:
    """Symmetric token F1 — the official LoCoMo paper metric (SQuAD formula).

    Applied between the *generated answer* and the gold answer.
    Unlike token recall, this penalises answers that add irrelevant tokens,
    matching the paper's methodology.
    """
    gold_toks = _answer_tokens(gold)
    pred_toks = _answer_tokens(predicted)
    if not gold_toks or not pred_toks:
        return 0.0
    common = gold_toks & pred_toks
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


def _score_generated(gold: str, generated: str) -> tuple[float, str]:
    """Score a generated answer vs gold with symmetric token F1."""
    return _token_f1(str(gold), str(generated)), _SCORE_TOKEN_F1


# ---------------------------------------------------------------------------
# Dialogue grouping
# ---------------------------------------------------------------------------


def _dialogue_id(case_id: str) -> str:
    """Extract the dialogue prefix from a LoCoMo case ID.

    "conv-26-q0001"    → "conv-26"
    "locomo-smoke-001" → "locomo-smoke-001"   (no -qNNNN suffix → own group)
    """
    idx = case_id.rfind("-q")
    if idx > 0 and case_id[idx + 2 :].isdigit():
        return case_id[:idx]
    return case_id


# ---------------------------------------------------------------------------
# Main batched runner
# ---------------------------------------------------------------------------


async def run_locomo_benchmark(
    client: GraphknowsMCPClient,
    options: FileBackedBenchmarkOptions,
) -> list[BenchmarkCaseResult]:
    """Run LoCoMo benchmark with per-dialogue session batching.

    Each unique dialogue's sessions are ingested ONCE, all questions for that
    dialogue are queried against the same session, then the session is purged.

    When ``options.generate_answers`` is True a DSPy generation step is added
    between retrieval and scoring, making scores paper-comparable (token F1
    on generated answer vs gold, matching arXiv 2402.17753).
    """
    # ── Optional pre-run LTM purge (eliminates cross-run contamination) ──────
    if options.pre_run_purge_ltm:
        log.info("Purging all LTM before eval run (pre_run_purge_ltm=True)")
        await client.call_tool("memory_purge", {"scope": "ltm", "session_id": ""})

    # ── Initialise answer generator (lazy — skipped when not requested) ───────
    generator = None
    if options.generate_answers:
        from evaluation.core.generation import AnswerGenerator

        generator = AnswerGenerator(model=options.generation_model or None)
        log.info("Answer generation enabled (paper-comparable token F1 scoring)")

    rows = _read_rows(options.data_root / options.questions_file)
    selected = rows[options.offset : options.offset + options.limit]

    # Group by dialogue, preserving question order within each dialogue.
    dialogue_groups: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        case_id = _first_string(row, ("id", "question_id", "case_id"), "")
        dlg = _dialogue_id(case_id)
        dialogue_groups.setdefault(dlg, []).append(row)

    cases: list[BenchmarkCaseResult] = []

    for dlg_id, dlg_rows in dialogue_groups.items():
        session_id = f"{options.session_prefix}-{options.benchmark}-{options.run_id}-{dlg_id}"

        # All questions in a dialogue embed the same sessions — read from first row.
        documents = _case_documents(
            dlg_rows[0],
            max_documents=options.max_documents,
            max_chars=options.max_chars_per_document,
        )

        ingest_outputs: list[dict[str, Any]] = []
        flush_result: dict[str, Any] = {}
        ltm_entity_list: list[dict[str, Any]] = []

        try:
            # ── Ingest all dialogue sessions once ───────────────────────────
            for idx, doc in enumerate(documents):
                result = await client.call_tool(
                    "memory_ingest",
                    {
                        "text": doc["text"],
                        "session_id": session_id,
                        "title": doc["title"] or f"{dlg_id}-doc-{idx}",
                    },
                )
                mcp_result = result if isinstance(result, dict) else {"result": result}
                ingest_outputs.append(
                    {
                        "document_title": doc["title"],
                        "is_error": "chunks" not in mcp_result,
                        **mcp_result,
                    }
                )

            # ── Flush to LTM (optional) ─────────────────────────────────────
            if options.promote_to_ltm:
                result = await client.call_tool("memory_flush", {"session_id": session_id})
                flush_result = result if isinstance(result, dict) else {"result": result}
                entity_result = await client.call_tool(
                    "ltm_entities", {"session_id": session_id, "limit": 500}
                )
                if isinstance(entity_result, dict):
                    ltm_entity_list = entity_result.get("entities", [])

            # ── Query — one per question in this dialogue ───────────────────
            ingest_elapsed_s = sum(float(r.get("elapsed_s", 0)) for r in ingest_outputs)
            flush_elapsed_s = float(flush_result.get("elapsed_s", 0)) if flush_result else 0.0

            for row in dlg_rows:
                case_id = _first_string(
                    row, ("id", "question_id", "case_id"), f"{options.benchmark}-unknown"
                )
                question = _first_string(row, ("question", "query", "prompt"))
                if not question:
                    raise ValueError(f"LoCoMo case {case_id} has no question/query/prompt")
                gold = row.get("answer", row.get("gold", row.get("target")))
                category = _first_string(row, ("category", "type", "task_type", "question_type"))

                result = await client.call_tool(
                    "memory_query",
                    {
                        "query": question,
                        "session_id": session_id,
                        "top_k": options.top_k,
                        "scope": options.recall_scope,
                    },
                )
                retrieval_raw = result if isinstance(result, dict) else {"result": result}
                recall = _parse_memory_query_hits(retrieval_raw)

                # ── Score: generation path (paper-comparable) or retrieval-only ──
                generated_answer: str | None = None
                if generator is not None:
                    generated_answer = await generator.generate(question, recall)
                    score, score_type = _score_generated(str(gold or ""), generated_answer)
                else:
                    score, score_type = _score_case(category, str(gold or ""), recall)

                ltm_graph_hit_count = sum(
                    1 for h in recall if "ltm_graph" in str(h.get("sources", ""))
                )

                cases.append(
                    BenchmarkCaseResult(
                        case_id=case_id,
                        question=question,
                        gold=gold,
                        memory_context=recall,
                        metadata={
                            "dataset": options.benchmark,
                            "category": category or None,
                            "dialogue_id": dlg_id,
                            "session_id": session_id,
                            "source_file": str(options.data_root / options.questions_file),
                            "relation_extraction_mode": options.relation_extraction_mode,
                            "recall_scope": options.recall_scope,
                            "promote_to_ltm": options.promote_to_ltm,
                            "top_k": options.top_k,
                            "ingest_documents": len(ingest_outputs),
                            "ingest_outputs": ingest_outputs,
                            "flush_result": flush_result or None,
                            "ltm_entity_list": ltm_entity_list,
                            "retrieval_raw": retrieval_raw,
                            "generated_answer": generated_answer,
                            "generate_answers": options.generate_answers,
                        },
                        metrics={
                            "recall_hit_count": len(recall),
                            "answer_support_score": score if score is not None else -1.0,
                            "answer_support_hit": score is not None and score >= 1.0,
                            "score_type": score_type,
                            "ingest_calls": len(ingest_outputs),
                            "ingest_error_count": sum(
                                1 for o in ingest_outputs if o.get("is_error", False)
                            ),
                            "ingested_chunks": sum(int(r.get("chunks", 0)) for r in ingest_outputs),
                            "ingested_entities": sum(
                                int(r.get("entities", 0)) for r in ingest_outputs
                            ),
                            "ingest_elapsed_s": round(ingest_elapsed_s, 3),
                            "flush_nodes": int(flush_result.get("nodes", 0)) if flush_result else 0,
                            "flush_edges": int(flush_result.get("edges", 0)) if flush_result else 0,
                            "flush_chunks": int(flush_result.get("chunks", 0))
                            if flush_result
                            else 0,
                            "flush_topics": int(flush_result.get("topics_created", 0))
                            if flush_result
                            else 0,
                            "flush_elapsed_s": round(flush_elapsed_s, 3),
                            "flush_error_count": len(flush_result.get("errors", []))
                            if flush_result
                            else 0,
                            "ltm_entity_count": len(ltm_entity_list),
                            "ltm_graph_hit_count": ltm_graph_hit_count,
                        },
                        score=score,
                    )
                )

        finally:
            if options.purge:
                await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return cases
