"""Mechanics shared by the retrieval channels.

Every channel that joins a symbolic tag back to chunks does the same two things:
drop the tags so common they carry no signal, then collapse the surviving rows
into one RRF-ranked score per chunk. Both were written twice —
``_session_chunk_count`` was byte-identical in ``frames.py`` and ``ontology.py``,
and the scoring tail differed only in a label.

Lives under ``channels/`` rather than ``symbolic/``: saturation is measured in
CHUNK coverage and RRF is a fusion concept, so both are retrieval concerns. The
layer contract puts ``channels`` above ``symbolic``, and pushing a retrieval idea
down into the knowledge layer would invert it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)


async def session_chunk_count(rt: Any, session_id: str, state: str | None) -> int:
    """Total CHUNK count for a session (optionally filtered by lifecycle state)."""
    if state:
        rows = await rt.store.query(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid AND c.state = $st RETURN count(c) AS c",
            sid=session_id,
            st=state,
        )
    else:
        rows = await rt.store.query(
            "MATCH (c:CHUNK) WHERE c.session_id = $sid RETURN count(c) AS c",
            sid=session_id,
        )
    return int(rows[0]["c"]) if rows else 0


async def tag_coverage(
    rt: Any,
    session_id: str,
    state: str | None,
    *,
    pattern: str,
    key: str,
) -> dict[str, int]:
    """How many of the session's chunks each tag covers, measured over the GRAPH.

    Over the graph, never over the rows a query happened to return. Counting the
    sample bounds the numerator by the page size while the denominator stays the
    whole session, so the ratio can never reach the threshold and the guard never
    fires — a saturation filter that runs, passes, and does nothing.

    *pattern* is the tag-to-chunk match with ``c`` bound to the CHUNK, and *key*
    the expression identifying the tag.
    """
    q = f"MATCH {pattern} WHERE c.session_id = $sid"
    if state:
        q += " AND c.state = $st"
    q += f" RETURN {key} AS k, count(DISTINCT c.chunk_id) AS n"
    rows = (
        await rt.store.query(q, sid=session_id, st=state)
        if state
        else await rt.store.query(q, sid=session_id)
    )
    return {r["k"]: int(r.get("n") or 0) for r in rows if r.get("k")}


def drop_saturated(
    rows: list[dict[str, Any]],
    *,
    field: str,
    coverage: dict[str, int],
    total: int,
    ratio: float,
    label: str,
) -> list[dict[str, Any]]:
    """Drop rows whose tag covers more than *ratio* of the session's chunks.

    A tag on nearly every chunk — ``Activity_ongoing``, or ``Person`` in a corpus
    of personal conversation — returns the whole conversation and discriminates
    nothing. Same failure mode, and same remedy, as saturated speaker entities.
    """
    if not total:
        return rows
    present = {str(r[field]) for r in rows if r.get(field)}
    saturated = {t for t in present if coverage.get(t, 0) / total > ratio}
    if not saturated:
        return rows
    log.debug("%s channel dropping saturated tags: %s", label, sorted(saturated))
    return [r for r in rows if str(r.get(field)) not in saturated]


def rank_chunks(rows: list[dict[str, Any]], top_k: int, source: str) -> dict[str, Any]:
    """Collapse tag-hit rows into one RRF-ranked entry per chunk.

    A chunk carrying several matching tags scores the sum of their match scores,
    so breadth of match counts before the ranks are handed to the fusion.
    """
    chunk_score: dict[str, float] = defaultdict(float)
    for r in rows:
        chunk_score[r["chunk_id"]] += float(r.get("score") or 0.0)

    ranked = sorted(chunk_score, key=lambda c: -chunk_score[c])[:top_k]
    return {
        cid: (
            rrf_score(rank),
            {"text": "", "entities": [], "sources": {source}, "doc_id": "", "metadata": {}},
        )
        for rank, cid in enumerate(ranked)
    }
