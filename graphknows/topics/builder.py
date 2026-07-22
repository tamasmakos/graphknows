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

    if cfg.source == "graph_embedding" and cfg.recompute_embeddings:
        # The only reason to touch the store here: node2vec parameters are the
        # granularity knob for this source, so a sweep must be able to move them.
        await store.compute_graph_embeddings(
            dimensions=cfg.n2v_dimensions,
            walk_length=cfg.n2v_walk_length,
            walks=cfg.n2v_walks,
        )

    entities = await _load_entities(store, cfg)
    if not entities:
        log.info("build_topics: no clusterable entities; nothing to do")
        return TopicSet(namespace=namespace, config=cfg.to_dict())

    labels = _cluster(entities, cfg)
    chunks = await _load_entity_chunks(store, [e["name"] for e in entities])
    topics, topic_texts, part = _assemble(entities, labels, chunks, cfg)
    await _describe(topics, topic_texts, cfg, summarizer)

    newman_q = metrics.newman_modularity(await _load_edges(store), part)
    totals = await _totals(store)
    grades = metrics.summarize_metrics(
        topics,
        total_chunks=totals["chunks"],
        total_entities=totals["entities"],
        newman_q=newman_q,
        native_modularity=await _native_modularity(store),
    )
    return TopicSet(topics=topics, metrics=grades, config=cfg.to_dict(), namespace=namespace)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


async def _load_entities(store: Any, cfg: TopicConfig) -> list[dict[str, Any]]:
    """Entities with the vector (or partition) this build clusters on."""
    if cfg.source == "community":
        rows = await store.query(
            "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
            "RETURN e.name AS name, e.pagerank AS pr, e.community_id AS cid"
        )
        return [r for r in rows if r.get("name")]
    rows = await store.query(
        "MATCH (e:ENTITY) WHERE e.graph_embedding IS NOT NULL "
        "RETURN e.name AS name, e.pagerank AS pr, e.graph_embedding AS vec"
    )
    out = [r for r in rows if r.get("name") and r.get("vec")]
    if not out:
        log.warning(
            "build_topics: no ENTITY.graph_embedding found — run a flush first, "
            "or use source='community'"
        )
        return out
    # node2vec skips entities with <3 relation edges, so on a sparse graph this
    # source can cover a small minority. Say so rather than silently producing
    # topics that describe 3% of the corpus.
    try:
        total = int((await store.query("MATCH (e:ENTITY) RETURN count(e) AS c"))[0]["c"])
    except Exception:
        total = 0
    if total and len(out) / total < 0.25:
        log.warning(
            "build_topics: source='graph_embedding' covers %d/%d entities (%.1f%%) — "
            "node2vec skips sparsely-connected nodes. Consider source='community'.",
            len(out),
            total,
            100.0 * len(out) / total,
        )
    return out


async def _load_entity_chunks(store: Any, names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Entity name -> [(chunk_id, text)] via MENTIONS."""
    rows = await store.query(
        "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
        "RETURN e.name AS name, c.chunk_id AS chunk_id, c.text AS text"
    )
    wanted = {n for n in names}
    out: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        name = r.get("name")
        if name in wanted and r.get("chunk_id"):
            out.setdefault(name, []).append((r["chunk_id"], r.get("text") or ""))
    return out


async def _load_edges(store: Any) -> list[tuple[str, str]]:
    """The REL edge list, for Newman modularity."""
    try:
        rows = await store.query(
            "MATCH (a:ENTITY)-[:REL]->(b:ENTITY) "
            "RETURN a.name_norm AS src, b.name_norm AS dst"
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


async def _native_modularity(store: Any) -> float | None:
    """ArcadeDB's score for the Louvain partition — reported, never optimised."""
    try:
        rows = await store.query(
            "CALL algo.modularityScore('community_id', ['REL']) "
            "YIELD modularity RETURN modularity AS m"
        )
        return float(rows[0]["m"]) if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _cluster(entities: list[dict[str, Any]], cfg: TopicConfig) -> list[int]:
    """Assign each entity a cluster label."""
    if cfg.source == "community":
        return [int(e.get("cid") or 0) for e in entities]

    vectors = _matrix(entities)
    if vectors.shape[0] < 2:
        return [0] * vectors.shape[0]
    # Average-linkage over cosine distance. scipy's implementation is C-level;
    # a naive Python agglomerative is O(N^3) and unusable at ~1000 entities.
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]
    from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

    distances = pdist(vectors, metric="cosine")
    linkage_matrix = linkage(distances, method="average")
    labels = fcluster(linkage_matrix, t=cfg.distance, criterion="distance")
    return [int(x) for x in labels]


def _matrix(entities: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([list(e["vec"]) for e in entities], dtype=np.float32)


# ---------------------------------------------------------------------------
# Assembly + description
# ---------------------------------------------------------------------------


def _assemble(
    entities: list[dict[str, Any]],
    labels: list[int],
    chunks: dict[str, list[tuple[str, str]]],
    cfg: TopicConfig,
) -> tuple[list[Topic], list[list[str]], dict[str, int]]:
    """Group clustered entities into Topics, dropping clusters below min_size.

    Returns ``(topics, member_texts_per_topic, entity_norm -> topic_index)``.
    The texts ride alongside rather than on the Topic because they are working
    state for description, not part of the record; the partition map is built
    here so modularity does not need an O(N*M) reverse lookup afterwards.
    """
    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)

    has_vectors = bool(entities) and "vec" in entities[0]
    vectors = _matrix(entities) if has_vectors else np.zeros((0, 0), dtype=np.float32)

    topics: list[Topic] = []
    topic_texts: list[list[str]] = []
    partition: dict[str, int] = {}
    for label, members in sorted(grouped.items()):
        if len(members) < cfg.min_size:
            continue
        ranked = sorted(
            ((str(entities[i]["name"]), float(entities[i].get("pr") or 0.0)) for i in members),
            key=lambda kv: -kv[1],
        )
        chunk_map: dict[str, str] = {}
        for name, _ in ranked:
            for cid, text in chunks.get(name, []):
                chunk_map.setdefault(cid, text)
        coh, centroid = metrics.coherence(vectors[members]) if has_vectors else (0.0, [])

        index = len(topics)
        for name, _ in ranked:
            partition[_norm(name)] = index
        topics.append(
            Topic(
                topic_id=f"topic:{cfg.fingerprint()}:{label}",
                title=" · ".join(n for n, _ in ranked[: cfg.max_title_entities]),
                entities=ranked,
                chunk_ids=list(chunk_map),
                coherence=round(coh, 4),
                embedding=centroid,
            )
        )
        topic_texts.append(list(chunk_map.values()))
    return topics, topic_texts, partition


async def _describe(
    topics: list[Topic], topic_texts: list[list[str]], cfg: TopicConfig, summarizer: Any
) -> None:
    """Attach the text-derived fields to each topic, in place.

    One embedding pass per topic yields three things: the TOPIC vector, the
    coherence, and the lead passage. The vector is not optional — ``TopicChannel``
    vector-searches ``TOPIC.embedding``, so a topic without one is written but
    unreachable, which is how a topic layer ends up looking built and inert.
    """
    background = metrics.token_sets(topic_texts)
    for i, topic in enumerate(topics):
        texts = topic_texts[i]
        others = background[:i] + background[i + 1 :]
        topic.keywords = metrics.keywords(texts, others, top_k=cfg.keywords_per_topic)

        centroid, coh, lead = _text_profile(texts)
        # Entity-vector coherence (graph_embedding source) is kept when present;
        # otherwise the text-derived one applies, so the metric is comparable.
        if centroid:
            topic.embedding = centroid
            if not topic.coherence:
                topic.coherence = round(coh, 4)
        if lead and lead in texts:
            topic.lead_chunk_id = topic.chunk_ids[texts.index(lead)]
        topic.summary = await _summary(topic, texts, lead, cfg, summarizer)


def _text_profile(texts: list[str]) -> tuple[list[float], float, str]:
    """``(centroid, coherence, lead_passage)`` from the member chunk texts."""
    usable = [t for t in texts if t and t.strip()][:64]
    if not usable:
        return [], 0.0, ""
    from graphknows.storage.embedder import embed

    try:
        mat = np.asarray(embed(usable), dtype=np.float32)
    except Exception as exc:
        log.warning("topic text profile failed (%s)", exc)
        return [], 0.0, usable[0]
    coh, centroid = metrics.coherence(mat)
    if not centroid:
        return [], 0.0, usable[0]
    rows = np.linalg.norm(mat, axis=1, keepdims=True)
    rows[rows == 0.0] = 1.0
    sims = (mat / rows) @ np.asarray(centroid, dtype=np.float32)
    return centroid, coh, usable[int(np.argmax(sims))]


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
