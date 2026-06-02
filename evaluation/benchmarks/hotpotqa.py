"""HotpotQA support-retrieval benchmark helpers for graphknows evaluation."""

from __future__ import annotations

from typing import Any

from evaluation.benchmarks.huggingface import fetch_dataset_rows
from evaluation.benchmarks.types import BenchmarkCaseResult
from evaluation.core.client import GraphknowsMCPClient

HOTPOTQA_DATASET = "hotpotqa/hotpot_qa"
HOTPOTQA_CONFIG = "distractor"
HOTPOTQA_SPLIT = "validation"


def _context_documents(row: dict[str, Any], *, max_docs: int, max_chars_per_doc: int) -> list[dict[str, str]]:
    context = row.get("context", {})
    titles = list(context.get("title", []))
    sentences = list(context.get("sentences", []))
    documents: list[dict[str, str]] = []
    for index, (title, sentence_list) in enumerate(zip(titles, sentences, strict=False)):
        text = " ".join(str(sentence).strip() for sentence in sentence_list if str(sentence).strip())
        if not text:
            continue
        documents.append(
            {
                "id": f"{row.get('id', 'hotpot')}-{index}",
                "title": f"hotpotqa-{title}",
                "source": "hotpotqa",
                "text": f"Title: {title}\n{text[:max_chars_per_doc]}",
            }
        )
        if len(documents) >= max_docs:
            break
    return documents


def score_hotpotqa_support(
    memory_context: list[dict[str, Any]],
    *,
    supporting_titles: list[str],
    answer: str,
) -> dict[str, Any]:
    recalled_text = "\n".join(str(hit.get("text", "")) for hit in memory_context).lower()
    unique_titles = [title for title in dict.fromkeys(supporting_titles) if title]
    matched_titles = [title for title in unique_titles if title.lower() in recalled_text]
    answer_support_hit = bool(answer.strip() and answer.strip().lower() in recalled_text)
    return {
        "answer_support_hit": answer_support_hit,
        "supporting_title_hit": bool(matched_titles),
        "supporting_title_recall_at_k": (
            len(matched_titles) / len(unique_titles) if unique_titles else 0.0
        ),
    }


async def fetch_hotpotqa_rows(offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
    return fetch_dataset_rows(
        dataset=HOTPOTQA_DATASET,
        config=HOTPOTQA_CONFIG,
        split=HOTPOTQA_SPLIT,
        offset=offset,
        length=limit,
    )


async def run_hotpotqa_small(
    client: GraphknowsMCPClient,
    *,
    offset: int,
    limit: int,
    run_id: str,
    top_k: int,
    max_docs: int,
    max_chars_per_doc: int,
    purge: bool = True,
) -> list[BenchmarkCaseResult]:
    rows, total_rows = await fetch_hotpotqa_rows(offset, limit)
    cases: list[BenchmarkCaseResult] = []
    for row_index, row in enumerate(rows):
        session_id = f"{run_id}-hotpotqa-{row.get('id', offset + row_index)}"
        documents = _context_documents(
            row,
            max_docs=max_docs,
            max_chars_per_doc=max_chars_per_doc,
        )
        recall: list[dict[str, Any]] = []
        ingest_results: list[dict[str, Any]] = []
        try:
            for document in documents:
                result = await client.call_tool(
                    "memory_ingest",
                    {
                        "text": document["text"],
                        "session_id": session_id,
                        "title": document["title"],
                    },
                )
                ingest_results.append(result if isinstance(result, dict) else {"result": result})
            result = await client.call_tool(
                "memory_recall",
                {
                    "query": str(row.get("question", "")),
                    "session_id": session_id,
                    "top_k": top_k,
                    "scope": "stm",
                },
            )
            recall = list(result.get("hits", []))
        finally:
            if purge:
                await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

        supporting = row.get("supporting_facts", {})
        supporting_titles = list(supporting.get("title", []))
        metrics = score_hotpotqa_support(
            recall,
            supporting_titles=supporting_titles,
            answer=str(row.get("answer", "")),
        )
        metrics["recall_hit_count"] = len(recall)
        cases.append(
            BenchmarkCaseResult(
                case_id=str(row.get("id", offset + row_index)),
                question=str(row.get("question", "")),
                gold=row.get("answer"),
                memory_context=recall,
                metadata={
                    "dataset": HOTPOTQA_DATASET,
                    "config": HOTPOTQA_CONFIG,
                    "split": HOTPOTQA_SPLIT,
                    "offset": offset + row_index,
                    "official_rows": total_rows,
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "context_docs_indexed": len(documents),
                    "session_id": session_id,
                    "ingest_results": ingest_results,
                },
                metrics={
                    **metrics,
                    "ingest_calls": len(ingest_results),
                    "ingested_chunks": sum(int(result.get("chunks", 0)) for result in ingest_results),
                    "ingested_entities": sum(int(result.get("entities", 0)) for result in ingest_results),
                },
            )
        )
    return cases
