"""LTM retrieval: recall_by_document, recall_by_dialog, recall_by_topic."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RecallResult:
    source_type: str
    text: str
    entity_names: list[str]
    provenance: dict
    score: float = 1.0


def recall_by_document(
    driver: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallResult]:
    """Find document chunks whose mentioned entities match the query.

    Strategy:
    1. Find Entity nodes whose name contains any word from query (case-insensitive)
    2. Traverse Entity <- MENTIONS <- Chunk <- HAS_CHUNK <- Document
    3. Return top_k chunks ranked by number of matched entities

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query string.
        top_k: Max results to return.
        session_filter: Optional session_id to scope results.

    Returns:
        List of RecallResult ordered by relevance score.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    if not words:
        return []

    try:
        cypher = """
            MATCH (e:Entity)
            WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
            MATCH (c:Chunk)-[:MENTIONS]->(e)
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            WITH d, c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS match_count
            ORDER BY match_count DESC
            LIMIT $top_k
            RETURN d.title AS title, d.url AS url, c.text AS text,
                   entities, toFloat(match_count) AS score
        """
        params: dict[str, Any] = {"words": words, "top_k": top_k}
        if session_filter:
            cypher = """
                MATCH (e:Entity)
                WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                MATCH (d:Document)-[:HAS_CHUNK]->(c)
                WHERE d.session_id = $session_filter
                WITH d, c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS match_count
                ORDER BY match_count DESC
                LIMIT $top_k
                RETURN d.title AS title, d.url AS url, c.text AS text,
                       entities, toFloat(match_count) AS score
            """
            params["session_filter"] = session_filter

        with driver.session() as s:
            rows = s.run(cypher, **params).data()

        results = []
        for r in rows:
            results.append(RecallResult(
                source_type="document",
                text=r["text"],
                entity_names=list(r["entities"]),
                provenance={
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                },
                score=r["score"],
            ))
        return results
    except Exception:
        log.exception("recall_by_document failed")
        return []


def recall_by_dialog(
    driver: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallResult]:
    """Find past utterances whose mentioned entities match the query.

    Strategy:
    1. Find Entity nodes matching query words
    2. Traverse Entity <- MENTIONS <- Utterance <- SAID <- Participant <- PARTICIPANT <- Dialog
    3. Return top_k utterances with session context

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query.
        top_k: Max results.
        session_filter: Optional session_id to filter.

    Returns:
        List of RecallResult with dialog provenance.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    if not words:
        return []

    try:
        cypher = """
            MATCH (e:Entity)
            WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
            MATCH (u:Utterance)-[:MENTIONS]->(e)
            MATCH (p:Participant)-[:SAID]->(u)
            MATCH (dia:Dialog)-[:PARTICIPANT]->(p)
            WITH dia, u, p, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS match_count
            ORDER BY match_count DESC, u.turn_index ASC
            LIMIT $top_k
            RETURN dia.session_id AS session_id, u.text AS text, u.turn_index AS turn_index,
                   p.role AS role, p.name AS participant, entities,
                   toFloat(match_count) AS score
        """
        params: dict[str, Any] = {"words": words, "top_k": top_k}
        if session_filter:
            cypher = """
                MATCH (e:Entity)
                WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
                MATCH (u:Utterance)-[:MENTIONS]->(e)
                MATCH (p:Participant)-[:SAID]->(u)
                MATCH (dia:Dialog)-[:PARTICIPANT]->(p)
                WHERE dia.session_id = $session_filter
                WITH dia, u, p, collect(DISTINCT e.name) AS entities,
                     count(DISTINCT e) AS match_count
                ORDER BY match_count DESC, u.turn_index ASC
                LIMIT $top_k
                RETURN dia.session_id AS session_id, u.text AS text, u.turn_index AS turn_index,
                       p.role AS role, p.name AS participant, entities,
                       toFloat(match_count) AS score
            """
            params["session_filter"] = session_filter

        with driver.session() as s:
            rows = s.run(cypher, **params).data()

        results = []
        for r in rows:
            results.append(RecallResult(
                source_type="dialog",
                text=r["text"],
                entity_names=list(r["entities"]),
                provenance={
                    "session_id": r.get("session_id", ""),
                    "turn_index": r.get("turn_index", 0),
                    "role": r.get("role", ""),
                    "participant": r.get("participant", ""),
                },
                score=r["score"],
            ))
        return results
    except Exception:
        log.exception("recall_by_dialog failed")
        return []


def recall_by_topic(
    driver: Any,
    query: str,
    top_k: int = 5,
) -> list[RecallResult]:
    """Find Topics whose label or member entities match the query.

    Strategy:
    1. Find Topic nodes where label contains query words
    2. Also find Topics whose member Entities match query words
    3. Return top_k Topics with entity samples and session provenance

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query.
        top_k: Max results.

    Returns:
        List of RecallResult with topic provenance.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    if not words:
        return []

    try:
        # Match via summary text (primary) or entity name (secondary)
        cypher = """
            MATCH (t:Topic)
            WHERE any(word IN $words WHERE toLower(coalesce(t.summary, '')) CONTAINS toLower(word))
            OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(t)
            WITH t, collect(e.name)[..8] AS entities, 1.0 AS score
            RETURN t.community_id AS topic_id,
                   coalesce(t.summary, 'Topic ' + toString(t.community_id)) AS label,
                   entities, t.entity_count AS size, score
            UNION
            MATCH (e:Entity)
            WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
            MATCH (e)-[:BELONGS_TO]->(t:Topic)
            OPTIONAL MATCH (e2:Entity)-[:BELONGS_TO]->(t)
            WITH t, collect(e2.name)[..8] AS entities, 0.8 AS score
            RETURN t.community_id AS topic_id,
                   coalesce(t.summary, 'Topic ' + toString(t.community_id)) AS label,
                   entities, t.entity_count AS size, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        with driver.session() as s:
            rows = s.run(cypher, words=words, top_k=top_k).data()

        seen: set[int] = set()
        results = []
        for r in rows:
            tid = r["topic_id"]
            if tid in seen:
                continue
            seen.add(tid)
            # Deduplicate entity names while preserving order
            ents: list[str] = []
            for e in (r.get("entities") or []):
                if e not in ents:
                    ents.append(e)
            results.append(RecallResult(
                source_type="topic",
                text=r["label"],
                entity_names=ents,
                provenance={
                    "topic_id": tid,
                    "size": r.get("size", 0),
                },
                score=r["score"],
            ))
        return results
    except Exception:
        log.exception("recall_by_topic failed")
        return []
