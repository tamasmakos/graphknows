"""Chunk ranking - BM25 + cosine hybrid for search result ranking."""
from __future__ import annotations

import math
import re
from typing import Any


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def bm25_score(
    query_terms: list[str],
    doc_text: str,
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a document given query terms."""
    words = doc_text.lower().split()
    dl = len(words)
    word_freq: dict[str, int] = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    score = 0.0
    for term in query_terms:
        freq = word_freq.get(term.lower(), 0)
        if freq == 0:
            continue
        idf = math.log((1 + 1) / (1 + 1) + 1)  # simplified, single-doc
        score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))
    return score


def rank_documents(query: str, documents: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    """Rank documents by BM25 relevance and return top-k with citation IDs."""
    if not documents:
        return []

    query_terms = re.sub(r"[^\w\s]", "", query.lower()).split()
    texts = [str(doc.get("text", "") or doc.get("snippet", ""))[:3000] for doc in documents]
    avg_dl = sum(len(text.split()) for text in texts) / max(len(texts), 1)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, (doc, text) in enumerate(zip(documents, texts)):
        score = bm25_score(query_terms, text, avg_dl)
        scored.append((score, index, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    result = []
    for rank, (score, _idx, doc) in enumerate(scored[:top_k], 1):
        enriched = {**doc, "citation_id": f"Doc-{rank}", "relevance_score": round(score, 4)}
        result.append(enriched)
    return result
