"""Legacy recall module — entity-graph-based recall over Neo4j LTM.

This module provides three entity-match recall functions used by the MCP tools:
  - recall_by_document  — document chunks ranked by matched entities
  - recall_by_dialog    — conversation turns ranked by matched entities
  - recall_by_topic     — topics ranked by entity membership

These functions use keyword-level entity extraction (no model load) to find
matching entities, then retrieve LTM graph nodes ranked by overlap count.

All functions return lists of :class:`RecallHit` objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecallHit:
    """A single recall result."""

    text: str
    entity_names: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def _extract_query_tokens(query: str) -> list[str]:
    """Extract meaningful tokens from a query for entity matching.

    Args:
        query: Natural language query string.

    Returns:
        List of lowercased, deduplicated word tokens (≥3 chars), order-preserving.
    """
    tokens = re.findall(r"\b\w{3,}\b", query.lower())
    return list(dict.fromkeys(tokens))


def recall_by_document(
    driver: Any,
    query: str,
    top_k: int = 5,
) -> list[RecallHit]:
    """Find document chunks whose mentioned entities match the query.

    Performs a keyword-level entity match against Neo4j ENTITY nodes,
    then retrieves CHUNK nodes connected via HAS_ENTITY edges, ranked
    by number of matched entities.

    Args:
        driver: Neo4j GraphDatabase driver (already open).
        query: Free-text search query.
        top_k: Max results to return (1-20).

    Returns:
        List of RecallHit ordered by descending entity overlap score.
    """
    tokens = _extract_query_tokens(query)
    if not tokens:
        return []

    top_k = max(1, min(top_k, 20))

    cypher = """
    UNWIND $tokens AS token
    MATCH (e:ENTITY)
    WHERE toLower(e.name) CONTAINS token
    WITH collect(DISTINCT e.name) AS matched_names, count(DISTINCT e) AS match_count
    MATCH (c:CHUNK)-[:HAS_ENTITY]->(e2:ENTITY)
    WHERE e2.name IN matched_names
    OPTIONAL MATCH (f:FILE)-[:HAS_CHUNK]->(c)
    WITH c, f, collect(DISTINCT e2.name) AS chunk_entities, count(DISTINCT e2) AS overlap
    ORDER BY overlap DESC
    LIMIT $top_k
    RETURN c.text AS text,
           f.id AS file_id,
           chunk_entities AS entities,
           overlap AS score
    """

    with driver.session() as s:
        rows = s.run(cypher, tokens=tokens, top_k=top_k).data()

    hits: list[RecallHit] = []
    for r in rows:
        hits.append(
            RecallHit(
                text=r.get("text") or "",
                entity_names=r.get("entities") or [],
                provenance={
                    "file_id": r.get("file_id"),
                },
                score=float(r.get("score") or 0),
            )
        )
    return hits


def recall_by_dialog(
    driver: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallHit]:
    """Find past conversation utterances whose mentioned entities match.

    Matches TURN nodes (dialog turns) by entity co-occurrence. Optionally
    scopes to a single session.

    Args:
        driver: Neo4j GraphDatabase driver (already open).
        query: Free-text search query.
        top_k: Max results to return (1-20).
        session_filter: Optional session_id to restrict results.

    Returns:
        List of RecallHit ordered by descending entity overlap score.
    """
    tokens = _extract_query_tokens(query)
    if not tokens:
        return []

    top_k = max(1, min(top_k, 20))

    session_clause = "WHERE t.session_id = $session_id" if session_filter else ""
    cypher = f"""
    UNWIND $tokens AS token
    MATCH (e:ENTITY)
    WHERE toLower(e.name) CONTAINS token
    WITH collect(DISTINCT e.name) AS matched_names
    MATCH (t:TURN)
    {session_clause}
    WHERE any(name IN matched_names WHERE toLower(t.text) CONTAINS toLower(name))
    WITH t,
         [name IN matched_names WHERE toLower(t.text) CONTAINS toLower(name)] AS found,
         size([name IN matched_names WHERE toLower(t.text) CONTAINS toLower(name)]) AS overlap
    ORDER BY overlap DESC
    LIMIT $top_k
    RETURN t.text AS text,
           t.session_id AS session_id,
           t.role AS role,
           found AS entities,
           overlap AS score
    """

    params: dict[str, Any] = {"tokens": tokens, "top_k": top_k}
    if session_filter:
        params["session_id"] = session_filter

    with driver.session() as s:
        rows = s.run(cypher, **params).data()

    hits: list[RecallHit] = []
    for r in rows:
        hits.append(
            RecallHit(
                text=r.get("text") or "",
                entity_names=r.get("entities") or [],
                provenance={
                    "session_id": r.get("session_id"),
                    "role": r.get("role"),
                },
                score=float(r.get("score") or 0),
            )
        )
    return hits


def recall_by_topic(
    driver: Any,
    query: str,
    top_k: int = 5,
) -> list[RecallHit]:
    """Find topics whose label or member entities match the query.

    Matches Topic nodes by their label or by entity membership. Topics
    are scored by number of matching entity members.

    Args:
        driver: Neo4j GraphDatabase driver (already open).
        query: Free-text search query.
        top_k: Max results to return (1-20).

    Returns:
        List of RecallHit ordered by descending entity overlap score.
    """
    tokens = _extract_query_tokens(query)
    if not tokens:
        return []

    top_k = max(1, min(top_k, 20))

    cypher = """
    UNWIND $tokens AS token
    MATCH (e)
    WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS token
    WITH collect(DISTINCT e.name) AS matched_names
    MATCH (t:TOPIC)
    OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
    WHERE e2.name IN matched_names
    WITH t,
         collect(DISTINCT e2.name) AS matched_entities,
         count(DISTINCT e2) AS overlap
    ORDER BY overlap DESC, t.entity_count DESC
    LIMIT $top_k
    RETURN t.id AS topic_id,
           t.label AS label,
           t.summary AS summary,
           matched_entities AS entities,
           overlap AS score
    """

    with driver.session() as s:
        rows = s.run(cypher, tokens=tokens, top_k=top_k).data()

    hits: list[RecallHit] = []
    token_set = set(tokens)
    for r in rows:
        entity_score = float(r.get("score") or 0)
        # Fall back to label when summary is absent
        summary = r.get("summary") or r.get("label") or ""
        entities = r.get("entities") or []

        # When entity overlap is zero, compute text-overlap bonus from summary
        if entity_score == 0:
            text_bonus = sum(1 for t in token_set if t in summary.lower()) if summary else 0
            score = float(text_bonus)
        else:
            score = entity_score

        # Skip hits with no entity overlap AND no text match signal
        if score == 0:
            continue

        hits.append(
            RecallHit(
                text=summary,
                entity_names=entities,
                provenance={"topic_id": r.get("topic_id")},
                score=score,
            )
        )
    return hits
