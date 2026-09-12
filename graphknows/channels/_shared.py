"""Mechanics shared by the retrieval channels.

Every channel that joins a symbolic tag back to segments does the same two
things: drop the tags so common they carry no signal, then collapse the
surviving rows into one RRF-ranked score per segment.

Lives under ``channels/`` rather than ``symbolic/``: saturation is measured in
SEGMENT coverage and RRF is a fusion concept, so both are retrieval concerns.
The layer contract puts ``channels`` above ``symbolic``, and pushing a retrieval
idea down into the knowledge layer would invert it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from graphknows.ranking.rrf import rrf_score
from graphknows.storage.arcadedb._sql import quoted_list

log = logging.getLogger(__name__)


async def segment_count(rt: Any, source_ids: list[str]) -> int:
    """Total SEGMENT count across the given sources."""
    rows = await rt.store.query(
        f"MATCH (g:SEGMENT) WHERE g.source_id IN {quoted_list(source_ids)} RETURN count(g) AS c"
    )
    return int(rows[0]["c"]) if rows else 0


async def tag_coverage(
    rt: Any,
    source_ids: list[str],
    *,
    pattern: str,
    key: str,
) -> dict[str, int]:
    """How many of the sources' segments each tag covers, measured over the GRAPH.

    Over the graph, never over the rows a query happened to return. Counting the
    sample bounds the numerator by the page size while the denominator stays the
    whole scope, so the ratio can never reach the threshold and the guard never
    fires — a saturation filter that runs, passes, and does nothing.

    *pattern* is the tag-to-segment match with ``g`` bound to the SEGMENT, and
    *key* the expression identifying the tag.
    """
    rows = await rt.store.query(
        f"MATCH {pattern} WHERE g.source_id IN {quoted_list(source_ids)} "
        f"RETURN {key} AS k, count(DISTINCT g.id) AS n"
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
    """Drop rows whose tag covers more than *ratio* of the scope's segments.

    A tag on nearly every segment — ``Activity_ongoing``, or ``Person`` in a
    corpus of personal conversation — returns the whole corpus and
    discriminates nothing.
    """
    if not total:
        return rows
    present = {str(r[field]) for r in rows if r.get(field)}
    saturated = {t for t in present if coverage.get(t, 0) / total > ratio}
    if not saturated:
        return rows
    log.debug("%s channel dropping saturated tags: %s", label, sorted(saturated))
    return [r for r in rows if str(r.get(field)) not in saturated]


def rank_segments(rows: list[dict[str, Any]], top_k: int, source: str) -> dict[str, Any]:
    """Collapse tag-hit rows into one RRF-ranked entry per segment.

    A segment carrying several matching tags scores the sum of their match
    scores, so breadth of match counts before the ranks are handed to the
    fusion.
    """
    segment_score: dict[str, float] = defaultdict(float)
    for r in rows:
        segment_score[r["id"]] += float(r.get("score") or 0.0)

    ranked = sorted(segment_score, key=lambda s: -segment_score[s])[:top_k]
    return {
        sid: (
            rrf_score(rank),
            {"text": "", "entities": [], "sources": {source}, "doc_id": "", "metadata": {}},
        )
        for rank, sid in enumerate(ranked)
    }
