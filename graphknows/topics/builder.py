"""The topic stage: cluster -> describe -> grade.

Deliberately a *stage*, not a step inside ``flush_memory``. Rebuilding topics at
different settings must not require re-running entity resolution, PageRank,
node2vec and consolidation — let alone a re-ingest. ``build_topics`` reads an
already-consolidated graph and computes; writing is a separate call.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from graphknows.settings import TopicMode
from graphknows.topics import metrics
from graphknows.topics.config import TopicConfig
from graphknows.topics.model import Topic, TopicSet

log = logging.getLogger(__name__)


async def build_topics(
    store: Any,
    config: TopicConfig | None = None,
    summarizer: Any = None,
    namespace: str = "",
) -> TopicSet:
    """Cluster the entity graph into topics and grade the result.

    Pure with respect to the store: it reads and computes, it does not write.
    Use :func:`graphknows.topics.persist.persist_topics` to store the result.
    """
    cfg = config or TopicConfig()
    if cfg.mode is TopicMode.none:
        return TopicSet(namespace=namespace, config=cfg.to_dict())

    entities = await _load_entities(store, cfg)
    if not entities:
        log.info("build_topics: no clusterable entities; nothing to do")
        return TopicSet(namespace=namespace, config=cfg.to_dict())

    labels = _cluster(entities)
    entity_chunks, chunk_meta = await _load_entity_chunks(store, [e["name"] for e in entities])
    topics, part = _assemble(entities, labels, entity_chunks, cfg)
    await _describe(topics, chunk_meta, cfg, summarizer)

    newman_q = metrics.newman_modularity(await _load_edges(store), part)
    totals = await _totals(store)
    grades = metrics.summarize_metrics(
        topics,
        total_chunks=totals["chunks"],
        total_entities=totals["entities"],
        newman_q=newman_q,
        native_modularity=None,
    )
    return TopicSet(topics=topics, metrics=grades, config=cfg.to_dict(), namespace=namespace)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


async def _load_entities(store: Any, cfg: TopicConfig) -> list[dict[str, Any]]:
    """Entities carrying the Louvain partition this build clusters on."""
    rows = await store.query(
        "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
        "RETURN e.name AS name, e.pagerank AS pr, e.community_id AS cid"
    )
    return [r for r in rows if r.get("name")]


async def _load_entity_chunks(
    store: Any, names: list[str]
) -> tuple[dict[str, list[str]], dict[str, tuple[str, list[float]]]]:
    """``(entity name -> [chunk_id], chunk_id -> (text, embedding))`` via MENTIONS.

    Fetches ``c.embedding`` alongside the text so the stage never re-embeds:
    the chunk vectors were computed at ingest and persisted, and a chunk belongs
    to many topics (measured on conv-26: 769 memberships over 63 distinct
    chunks, a 12x redundancy). Re-embedding per topic made description 337s of a
    337.4s stage; reading the stored vectors makes it free.
    """
    rows = await store.query(
        "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
        "RETURN e.name AS name, c.chunk_id AS chunk_id, c.text AS text, "
        "c.embedding AS embedding"
    )
    wanted = set(names)
    by_entity: dict[str, list[str]] = {}
    chunk_meta: dict[str, tuple[str, list[float]]] = {}
    for r in rows:
        name, cid = r.get("name"), r.get("chunk_id")
        if name not in wanted or not cid:
            continue
        by_entity.setdefault(name, []).append(cid)
        if cid not in chunk_meta:
            chunk_meta[cid] = (r.get("text") or "", list(r.get("embedding") or []))
    return by_entity, chunk_meta


async def _load_edges(store: Any) -> list[tuple[str, str]]:
    """The REL edge list, for Newman modularity."""
    try:
        rows = await store.query(
            "MATCH (a:ENTITY)-[:REL]->(b:ENTITY) RETURN a.name_norm AS src, b.name_norm AS dst"
        )
    except Exception as exc:
        log.warning("build_topics: edge load failed (%s); modularity unavailable", exc)
        return []
    return [(r["src"], r["dst"]) for r in rows if r.get("src") and r.get("dst")]


async def _totals(store: Any) -> dict[str, int]:
    out = {"chunks": 0, "entities": 0}
    for key, q in (
        ("chunks", "MATCH (c:CHUNK) RETURN count(c) AS c"),
        ("entities", "MATCH (e:ENTITY) RETURN count(e) AS c"),
    ):
        try:
            rows = await store.query(q)
            out[key] = int(rows[0]["c"]) if rows else 0
        except Exception:
            out[key] = 0
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster(entities: list[dict[str, Any]]) -> list[int]:
    """Assign each entity a cluster label — the Louvain community it landed in."""
    return [int(e.get("cid") or 0) for e in entities]


# ---------------------------------------------------------------------------
# Assembly + description
# ---------------------------------------------------------------------------


def _assemble(
    entities: list[dict[str, Any]],
    labels: list[int],
    entity_chunks: dict[str, list[str]],
    cfg: TopicConfig,
) -> tuple[list[Topic], dict[str, int]]:
    """Group clustered entities into Topics, dropping clusters below min_size.

    Returns ``(topics, entity_norm -> topic_index)``. The partition map is built
    here so modularity does not need an O(N*M) reverse lookup afterwards; member
    texts are not carried, because ``chunk_ids`` plus the chunk map is enough.
    """
    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)

    topics: list[Topic] = []
    partition: dict[str, int] = {}
    for label, members in sorted(grouped.items()):
        if len(members) < cfg.min_size:
            continue
        ranked = sorted(
            ((str(entities[i]["name"]), float(entities[i].get("pr") or 0.0)) for i in members),
            key=lambda kv: -kv[1],
        )
        chunk_ids: dict[str, None] = {}  # ordered set
        for name, _ in ranked:
            for cid in entity_chunks.get(name, ()):
                chunk_ids.setdefault(cid, None)
        if not chunk_ids:
            # A cluster with no evidence chunks is not a topic: it can carry no
            # summary, no keywords and no centroid, so it would be written and
            # then be unreachable. These are typically date/time entities minted
            # as relation endpoints, which never get a MENTIONS edge.
            log.debug(
                "topic stage: dropping chunkless cluster %s (%d entities)", label, len(members)
            )
            continue
        index = len(topics)
        for name, _ in ranked:
            partition[_norm(name)] = index
        topics.append(
            Topic(
                topic_id=f"topic:{cfg.fingerprint()}:{label}",
                title=" · ".join(n for n, _ in ranked[: cfg.max_title_entities]),
                entities=ranked,
                chunk_ids=list(chunk_ids),
            )
        )
    return topics, partition


async def _describe(
    topics: list[Topic],
    chunk_meta: dict[str, tuple[str, list[float]]],
    cfg: TopicConfig,
    summarizer: Any,
) -> None:
    """Attach the text-derived fields to each topic, in place.

    Uses the chunk embeddings persisted at ingest — no embedding call. The
    centroid is computed but not persisted: ``coherence`` is measured against it
    and ``lead_chunk_id`` is the member nearest it, and nothing reads a stored
    ``TOPIC.embedding`` now that the topic retrieval channel is gone.
    """
    texts_per_topic = [[chunk_meta.get(cid, ("", []))[0] for cid in t.chunk_ids] for t in topics]
    background = metrics.token_sets(texts_per_topic)
    for i, topic in enumerate(topics):
        texts = texts_per_topic[i]
        others = background[:i] + background[i + 1 :]
        topic.keywords = metrics.keywords(texts, others, top_k=cfg.keywords_per_topic)

        centroid, coh, lead_idx = _profile(topic.chunk_ids, chunk_meta)
        # Text-derived coherence: mean cosine of member chunks to the centroid.
        if centroid:
            topic.embedding = centroid
            if not topic.coherence:
                topic.coherence = round(coh, 4)
        lead = ""
        if lead_idx is not None:
            topic.lead_chunk_id = topic.chunk_ids[lead_idx]
            lead = texts[lead_idx]
        topic.summary = await _summary(topic, texts, lead, cfg, summarizer)


def _profile(
    chunk_ids: list[str], chunk_meta: dict[str, tuple[str, list[float]]]
) -> tuple[list[float], float, int | None]:
    """``(centroid, coherence, index of the centroid-nearest member)``.

    Built entirely from the stored chunk vectors, so a topic costs no embedding
    call regardless of how many members it has.
    """
    rows: list[list[float]] = []
    positions: list[int] = []
    for i, cid in enumerate(chunk_ids):
        vec = chunk_meta.get(cid, ("", []))[1]
        if vec:
            rows.append(vec)
            positions.append(i)
    if not rows:
        return [], 0.0, None
    mat = np.asarray(rows, dtype=np.float32)
    coh, centroid = metrics.coherence(mat)
    if not centroid:
        return [], 0.0, positions[0]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    sims = (mat / norms) @ np.asarray(centroid, dtype=np.float32)
    return centroid, coh, positions[int(np.argmax(sims))]


async def _summary(
    topic: Topic, texts: list[str], lead: str, cfg: TopicConfig, summarizer: Any
) -> str:
    if cfg.mode is TopicMode.llm and summarizer is not None:
        try:
            out = await summarizer.summarize(
                members=[n for n, _ in topic.entities[: cfg.max_title_entities]],
                passages=[t for t in texts if t.strip()][: cfg.max_passages],
                fallback_title=topic.title,
            )
            if out:
                return str(out)
        except Exception as exc:
            log.warning("topic summarizer failed (%s); using extractive", exc)
    return f"{topic.title}. {lead[: cfg.summary_max_chars]}".strip()


def _norm(name: str) -> str:
    """Match GraphStore's ENTITY.name_norm so the edge list joins the partition."""
    return " ".join(name.split()).casefold()


__all__ = ["build_topics"]
