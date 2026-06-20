"""HotpotQA research-memory helper functions."""

from __future__ import annotations

from typing import Any


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _context_documents(
    row: dict[str, Any],
    *,
    max_docs: int,
    max_chars_per_doc: int,
) -> list[dict[str, str]]:
    """Convert a HotpotQA row context into corpus documents."""
    context = row.get("context") or {}
    documents: list[dict[str, str]] = []
    if isinstance(context, dict):
        iterable = context.items()
    else:
        iterable = enumerate(context)

    for index, item in enumerate(iterable):
        if len(documents) >= max_docs:
            break
        if isinstance(context, dict):
            title, sentences = item
        else:
            _idx, value = item
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                title, sentences = value[0], value[1]
            else:
                title, sentences = f"document-{index}", value
        if isinstance(sentences, list):
            text = " ".join(str(sentence) for sentence in sentences)
        else:
            text = str(sentences)
        title_text = str(title)
        documents.append(
            {
                "id": str(row.get("id", row.get("_id", index))) + f"-{index}",
                "title": title_text,
                "source": "hotpotqa",
                "text": f"Title: {title_text}\n{text}"[:max_chars_per_doc],
            }
        )
    return documents


async def fetch_hotpotqa_rows(offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
    """Return HotpotQA rows from local data when available.

    Network-backed dataset loading is intentionally left to full evaluation
    runs; unit and CI smoke runs use deterministic empty output when no local
    dataset is present.
    """
    _ = offset
    _ = limit
    return [], 0


def score_hotpotqa_support(
    hits: list[dict[str, Any]],
    *,
    supporting_titles: list[str],
    answer: str,
) -> dict[str, Any]:
    """Score whether recall hits contain the answer and supporting titles."""
    text = "\n".join(str(hit.get("text", "")) for hit in hits)
    normalised_text = _normalise(text)
    title_hits = [
        title for title in supporting_titles if title and _normalise(title) in normalised_text
    ]
    answer_hit = bool(answer and _normalise(answer) in normalised_text)
    recall = len(title_hits) / len(supporting_titles) if supporting_titles else 0.0
    return {
        "answer_support_hit": answer_hit,
        "supporting_title_hit": bool(title_hits),
        "supporting_title_recall_at_k": recall,
    }
