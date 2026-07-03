"""Fact-lookup retrieval: a first-class, subject-scoped, confidence-gated channel.

The graph audit showed that clean directional facts (``subject -RELATION-> object``)
are wired to their source CHUNK via ``chunk_id`` (Stage 1) but were never *read* by
retrieval. This channel closes that gap:

1. Detect which of the session's fact *subjects* the query names (subject-scoped).
2. Fetch those subjects' wired fact edges (each carries ``chunk_id`` + evidence).
3. Rank facts by embedding similarity of the ``subject relation object`` statement
   to the query.
4. **Gate**: emit nothing unless a subject matched *and* the best fact clears a
   similarity threshold — so the channel is silent (clean fallback to vector) on
   queries it cannot help, rather than diluting the pool (naive fusion regressed in
   the audit).

The channel is deliberately isolated: it returns its own ranked hits, each pointing
at a source chunk and carrying the fact triple + evidence, for the caller to route
ahead of the vector pool rather than blindly fuse.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import numpy as np

from graphknows.memory.stores.embedder import embed

log = logging.getLogger(__name__)

# Similarity floor for a fact to count as a confident answer. Below this the
# channel stays silent so it never displaces vector hits on queries it can't help.
DEFAULT_MIN_SIM = 0.30
# Cap on surfaced hits — enough to cover list-aggregation answers without flooding.
DEFAULT_MAX_HITS = 10

_word = re.compile(r"[a-z0-9']+")


def _match_subjects(query: str, subjects: list[str]) -> list[str]:
    """Return the fact subjects the query names (case-insensitive, word-boundary)."""
    q_lower = query.lower()
    matched: list[str] = []
    for subj in subjects:
        s = (subj or "").strip()
        if len(s) < 3:
            continue
        if re.search(rf"\b{re.escape(s.lower())}\b", q_lower):
            matched.append(subj)
    return matched


def _cosine(query_emb: list[float], mat: np.ndarray) -> np.ndarray:
    q = np.asarray(query_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return m @ q


async def fact_lookup(
    mg: Any,
    query: str,
    session_id: str,
    query_emb: list[float],
    embed_model: str = "",
    *,
    min_sim: float = DEFAULT_MIN_SIM,
    max_hits: int = DEFAULT_MAX_HITS,
) -> list[dict[str, Any]]:
    """Return confidence-gated fact hits for *query*, or ``[]`` when the gate fails.

    Each hit: ``{chunk_id, subject, relation, object, evidence, score}``.
    """
    if not session_id or not query_emb:
        return []

    subjects = await mg.query_fact_subjects(session_id)
    matched = _match_subjects(query, subjects)
    if not matched:
        return []  # gate: no subject anchor in the query

    facts = await mg.query_facts_by_subjects(session_id, matched, top_k=200)
    if not facts:
        return []

    statements = [
        f"{f['subject']} {f['relation'].replace('_', ' ').lower()} {f['object']}"
        for f in facts
    ]
    embs = await asyncio.to_thread(embed, statements, embed_model)
    sims = _cosine(query_emb, np.asarray(embs))

    if float(sims.max()) < min_sim:
        return []  # gate: no fact is a confident answer

    # Keep the best-scoring fact per source chunk, above the floor.
    best: dict[str, dict[str, Any]] = {}
    for f, sim in zip(facts, sims.tolist(), strict=False):
        if sim < min_sim:
            continue
        cid = f["chunk_id"]
        if cid not in best or sim > best[cid]["score"]:
            best[cid] = {**f, "score": float(sim)}

    hits = sorted(best.values(), key=lambda h: -h["score"])[:max_hits]
    log.debug(
        "fact_lookup session=%s subjects=%s facts=%d hits=%d top_sim=%.3f",
        session_id,
        matched,
        len(facts),
        len(hits),
        float(sims.max()),
    )
    return hits
