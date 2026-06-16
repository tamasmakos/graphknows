"""LTMEnrichmentService — GDS enrichment with Python-native fallback.

When Neo4j GDS plugin is unavailable (gds_unavailable), a fully in-process
pipeline runs instead:
  1. Read entity graph from Neo4j (igraph).
  2. Biased random walks → co-occurrence SVD → node2vec-style embeddings.
  3. Cosine-similarity edge weights (structural proximity as Leiden signal).
  4. PageRank + betweenness via igraph.
  5. Leiden community detection (leidenalg) with weighted edges.
  6. Write community_id / pagerank / betweenness back to Neo4j ENTITY nodes.
  7. Update relationship weights in Neo4j.

This mirrors what the GDS plugin does but runs entirely in Python, so the
rest of the pipeline (community metadata → topic summarization) is unaffected
by whether GDS is installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import igraph
import leidenalg
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from graphknows.memory.models.gds import CentralityResult, CommunityResult, GDSResult

log = logging.getLogger("graphknows.memory.ltm.enrichment")

PROJECTED_GRAPH_NAME = "ltm-entities"
_DROP_GRAPH_CQL = "CALL gds.graph.drop($name, false) YIELD graphName"


# ---------------------------------------------------------------------------
# GDS availability check
# ---------------------------------------------------------------------------


def _gds_available(s: Any) -> bool:
    try:
        row = s.run(
            "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gds.' RETURN count(name) AS c"
        ).single()
    except Exception as exc:
        if "gds." in str(exc) and ("ProcedureNotFound" in str(exc) or "Unknown procedure" in str(exc)):
            return False
        log.warning("Could not inspect GDS procedures; treating as unavailable: %s", exc)
        return False
    return bool(row and row["c"] > 0)


# ---------------------------------------------------------------------------
# Python-native pipeline (runs when GDS is not installed)
# ---------------------------------------------------------------------------


def _biased_random_walk_embeddings(
    g: igraph.Graph,
    dim: int = 64,
    walk_length: int = 30,
    walks_per_node: int = 10,
    window: int = 5,
) -> np.ndarray | None:
    """Node2Vec-inspired: biased random walks → co-occurrence → SVD.

    Nodes that co-appear frequently in walks (because they share heavy
    edges) end up with similar embeddings. Cosine similarity of these
    embeddings is then used to up-weight Leiden edges between structurally
    similar entities, improving modularity.
    """
    n = g.vcount()
    if n < 4:
        return None

    # Pre-compute weighted neighbour lists for fast sampling
    adjacency: list[list[int]] = [[] for _ in range(n)]
    adj_weights: list[list[float]] = [[] for _ in range(n)]
    for e in g.es:
        u, v = e.source, e.target
        w = float(e["weight"] if "weight" in g.es.attributes() else 1.0)
        w = max(w, 1e-6)
        adjacency[u].append(v)
        adj_weights[u].append(w)
        if not g.is_directed():
            adjacency[v].append(u)
            adj_weights[v].append(w)

    cum_weights: list[list[float]] = []
    for ws in adj_weights:
        cum: list[float] = []
        total = 0.0
        for w in ws:
            total += w
            cum.append(total)
        cum_weights.append(cum)

    def _weighted_choice(v: int) -> int | None:
        nbrs = adjacency[v]
        if not nbrs:
            return None
        cum = cum_weights[v]
        r = random.uniform(0, cum[-1])
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return nbrs[lo]

    # Generate walks
    node_order = list(range(n))
    cooc = np.zeros((n, n), dtype=np.float32)
    for _ in range(walks_per_node):
        random.shuffle(node_order)
        for start in node_order:
            walk = [start]
            for _ in range(walk_length - 1):
                nxt = _weighted_choice(walk[-1])
                if nxt is None:
                    break
                walk.append(nxt)
            # Sliding window co-occurrence (symmetric)
            for i, u in enumerate(walk):
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    v = walk[j]
                    if u != v:
                        cooc[u, v] += 1.0

    if cooc.sum() == 0:
        return None

    actual_dim = min(dim, n - 1, 64)
    if actual_dim < 2:
        return None

    try:
        svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        emb = svd.fit_transform(cooc)
    except Exception as exc:
        log.warning("SVD embedding failed: %s", exc)
        return None

    # L2-normalize to unit vectors so dot product == cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-9, norms, 1.0)
    emb = emb / safe_norms
    return emb


def _run_python_native_pipeline(driver: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Python-native GDS substitute using igraph + leidenalg.

    Reads the entity graph from Neo4j, runs the full enrichment pipeline
    in-process, and writes results back to Neo4j.
    """
    t0 = time.time()
    with driver.session() as s:
        entity_rows = s.run(
            "MATCH (e:ENTITY) RETURN e.name AS name"
        ).data()
        rel_rows = s.run(
            "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
            "RETURN a.name AS src, b.name AS tgt, coalesce(r.weight, 1.0) AS weight"
        ).data()

    if not entity_rows:
        result["error"] = "no_entities"
        return result

    n = len(entity_rows)
    result["entity_count"] = n
    result["edge_count"] = len(rel_rows)

    if n < 3:
        result["error"] = f"too_small:{n}"
        return result

    # ── Build igraph (undirected for community detection) ──────────────────
    name_to_idx = {r["name"]: i for i, r in enumerate(entity_rows)}

    g = igraph.Graph(n=n, directed=False)
    g.vs["name"] = [r["name"] for r in entity_rows]

    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    seen_edges: set[tuple[int, int]] = set()

    for r in rel_rows:
        src = name_to_idx.get(r["src"])
        tgt = name_to_idx.get(r["tgt"])
        if src is None or tgt is None or src == tgt:
            continue
        key = (min(src, tgt), max(src, tgt))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_pairs.append((src, tgt))
        edge_weights.append(max(float(r["weight"]), 0.01))

    if not edge_pairs:
        result["error"] = "no_edges"
        return result

    g.add_edges(edge_pairs)
    g.es["weight"] = edge_weights

    # ── PageRank ───────────────────────────────────────────────────────────
    pagerank = g.pagerank(weights="weight", damping=0.85)

    # ── Betweenness (normalized) ───────────────────────────────────────────
    betweenness = g.betweenness(weights="weight", directed=False)
    max_betw = max(betweenness) if betweenness else 1.0
    betweenness_norm = [b / max_betw if max_betw > 1e-12 else 0.0 for b in betweenness]

    # ── Node2Vec-style embeddings → cosine edge weights ────────────────────
    log.info("Running random-walk SVD embeddings on %d entities, %d edges", n, len(edge_pairs))
    embeddings = _biased_random_walk_embeddings(
        g, dim=64, walk_length=30, walks_per_node=10, window=5
    )

    leiden_weights: list[float] = list(edge_weights)  # default: original weights
    if embeddings is not None:
        for i, (u, v) in enumerate(edge_pairs):
            eu, ev = embeddings[u], embeddings[v]
            cos_sim = float(np.dot(eu, ev))  # unit vectors → cosine similarity
            # Scale cosine [-1,1] → [0.01, 2.0] then multiply original weight
            structural_w = max(0.01, (cos_sim + 1.0))  # in [0.01, 2.0]
            leiden_weights[i] = edge_weights[i] * structural_w
        g.es["weight"] = leiden_weights
        log.info("Edge weights updated with cosine similarity (node2vec embeddings)")
    else:
        log.warning("Embedding failed — using original edge weights for Leiden")

    # ── Leiden community detection ─────────────────────────────────────────
    try:
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            n_iterations=10,
            seed=42,
        )
        modularity = float(partition.quality())
        membership = partition.membership
        n_communities = len(set(membership))
    except Exception as exc:
        log.error("Leiden failed: %s", exc)
        result["error"] = f"leiden:{exc}"
        return result

    result["community_count"] = n_communities
    result["modularity"] = modularity
    result["algorithm_used"] = "leiden_python_native"
    result["did_converge"] = True
    result["ran_iterations"] = 10

    log.info(
        "Leiden: %d communities, modularity=%.4f in %.0fms",
        n_communities,
        modularity,
        (time.time() - t0) * 1000,
    )

    # ── Write results back to Neo4j ────────────────────────────────────────
    with driver.session() as s:
        for i, r in enumerate(entity_rows):
            s.run(
                "MATCH (e:ENTITY {name: $name}) "
                "SET e.community_id      = $cid, "
                "    e.pagerank          = $pr, "
                "    e.betweenness       = $betw, "
                "    e.degree_centrality = $deg",
                name=r["name"],
                cid=int(membership[i]),
                pr=round(float(pagerank[i]), 6),
                betw=round(float(betweenness_norm[i]), 6),
                deg=round(float(g.degree(i)) / max(1, n - 1), 6),
            ).consume()

        # Update relationship weights (cosine-boosted)
        for i, (src_idx, tgt_idx) in enumerate(edge_pairs):
            src_name = entity_rows[src_idx]["name"]
            tgt_name = entity_rows[tgt_idx]["name"]
            new_w = round(float(leiden_weights[i]), 6)
            s.run(
                "MATCH (a:ENTITY {name: $src})-[r]->(b:ENTITY {name: $tgt}) "
                "SET r.weight = $w",
                src=src_name,
                tgt=tgt_name,
                w=new_w,
            ).consume()

    log.info(
        "Python-native GDS: wrote %d community_id + pagerank to ENTITY nodes in %.1fs",
        n,
        time.time() - t0,
    )

    # Track changed community IDs for topic refresh
    result["changed_community_ids"] = list(range(n_communities))
    return result


# ---------------------------------------------------------------------------
# GDS procedures (when plugin IS available)
# ---------------------------------------------------------------------------


def _project_initial_graph(s: Any, ec: int, edge_count: int) -> None:
    with contextlib.suppress(Exception):
        s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME).consume()
    try:
        s.run(
            """
            CALL gds.graph.project(
                'ltm-entities', 'ENTITY',
                {all: {type: '*', orientation: 'NATURAL',
                       properties: {weight: {defaultValue: 1.0}}}}
            ) YIELD graphName, nodeCount, relationshipCount
            """,
        ).consume()
    except Exception:
        # Fallback to Cypher projection for older GDS
        s.run(
            """
            CALL gds.graph.project.cypher(
                'ltm-entities',
                'MATCH (n:ENTITY) RETURN id(n) AS id',
                'MATCH (a:ENTITY)-[r]->(b:ENTITY) '
                'RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
            ) YIELD graphName, nodeCount, relationshipCount
            """,
        ).consume()
    log.info("Projected graph 'ltm-entities': %d entities, %d edges", ec, edge_count)


def _project_undirected(s: Any) -> None:
    with contextlib.suppress(Exception):
        s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME).consume()
    try:
        s.run(
            """
            CALL gds.graph.project(
                'ltm-entities', 'ENTITY',
                {all: {type: '*', orientation: 'UNDIRECTED',
                       properties: {weight: {defaultValue: 1.0}}}}
            ) YIELD graphName
            """,
        ).consume()
    except Exception:
        s.run(
            """
            CALL gds.graph.project.cypher(
                'ltm-entities',
                'MATCH (n:ENTITY) RETURN id(n) AS id',
                'MATCH (a:ENTITY)-[r]->(b:ENTITY)
                 RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight
                 UNION
                 MATCH (a:ENTITY)<-[r]-(b:ENTITY)
                 RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
            ) YIELD graphName
            """,
        ).consume()


def _run_node2vec(s: Any) -> None:
    try:
        res = s.run(
            """
            CALL gds.node2vec.write('ltm-entities', {
                writeProperty: 'node2vec',
                embeddingDimension: 64,
                walkLength: 30,
                walksPerNode: 10,
                inOutFactor: 1.0,
                returnFactor: 1.0,
                iterations: 5
            }) YIELD nodePropertiesWritten, computeMillis
            """,
        ).single()
        if res:
            log.info("GDS Node2Vec: %d embeddings in %dms", res["nodePropertiesWritten"], res["computeMillis"])
    except Exception as exc:
        log.warning("GDS Node2Vec failed (non-fatal): %s", exc)


def _run_cosine_weights(s: Any) -> None:
    try:
        s.run(
            """
            MATCH (a:ENTITY)-[r]->(b:ENTITY)
            WHERE a.node2vec IS NOT NULL AND b.node2vec IS NOT NULL
            SET r.weight = gds.similarity.cosine(a.node2vec, b.node2vec)
            """
        ).consume()
    except Exception as exc:
        log.warning("GDS cosine weight update failed (non-fatal): %s", exc)


def _run_pagerank(s: Any) -> None:
    try:
        s.run(
            """
            CALL gds.pageRank.write('ltm-entities', {
                writeProperty: 'pagerank', maxIterations: 20,
                dampingFactor: 0.85, relationshipWeightProperty: 'weight'
            }) YIELD nodePropertiesWritten
            """,
        ).consume()
    except Exception:
        try:
            s.run(
                """
                CALL gds.pageRank.write('ltm-entities', {
                    writeProperty: 'pagerank', maxIterations: 20, dampingFactor: 0.85
                }) YIELD nodePropertiesWritten
                """,
            ).consume()
        except Exception as exc2:
            log.warning("GDS PageRank failed (non-fatal): %s", exc2)


def _run_degree_centrality(s: Any) -> None:
    try:
        s.run(
            "CALL gds.degree.write('ltm-entities', {writeProperty: 'degree_centrality'})"
            " YIELD nodePropertiesWritten"
        ).consume()
    except Exception as exc:
        log.warning("GDS degree centrality failed (non-fatal): %s", exc)


def _run_betweenness(s: Any, ec: int) -> None:
    if ec > 5000:
        return
    try:
        sampling = max(1, min(int(ec * 0.1), 100))
        s.run(
            "CALL gds.betweenness.write('ltm-entities', "
            "{writeProperty: 'betweenness', samplingSize: $ss}) YIELD nodePropertiesWritten",
            ss=sampling,
        ).consume()
    except Exception as exc:
        log.warning("GDS betweenness failed (non-fatal): %s", exc)


_COMMUNITY_VARIANTS: list[tuple[str, str]] = [
    (
        "leiden_weighted",
        """
        CALL gds.leiden.write('ltm-entities', {
            writeProperty: 'community_id',
            relationshipWeightProperty: 'weight'
        }) YIELD communityCount, modularity, modularities, ranLevels
        """,
    ),
    (
        "leiden_unweighted",
        "CALL gds.leiden.write('ltm-entities', {writeProperty: 'community_id'})"
        " YIELD communityCount, modularity, modularities, ranLevels",
    ),
    (
        "louvain_weighted",
        """
        CALL gds.louvain.write('ltm-entities', {
            writeProperty: 'community_id',
            relationshipWeightProperty: 'weight'
        }) YIELD communityCount, modularity, modularities, ranLevels
        """,
    ),
    (
        "louvain_unweighted",
        "CALL gds.louvain.write('ltm-entities', {writeProperty: 'community_id'})"
        " YIELD communityCount, modularity, modularities, ranLevels",
    ),
]


def _run_community_detection(s: Any) -> tuple[Any, str]:
    for name, cypher in _COMMUNITY_VARIANTS:
        try:
            result = s.run(cypher).single()
            return result, name
        except Exception as exc:
            log.warning("GDS community %s failed: %s", name, exc)
    log.error("All GDS community detection variants failed")
    return None, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_feature_engineering(driver: Any, session_id: str = "") -> dict[str, Any]:
    """Run full GDS enrichment pipeline, falling back to Python-native when GDS unavailable.

    Either path writes community_id + pagerank + betweenness to Neo4j ENTITY nodes
    and updates relationship weights with cosine-similarity-boosted scores.
    """
    t0 = time.time()
    result: dict[str, Any] = {
        "entity_count": 0,
        "edge_count": 0,
        "community_count": 0,
        "modularity": 0.0,
        "modularities": [],
        "ran_iterations": 0,
        "did_converge": False,
        "algorithm_used": "",
        "changed_community_ids": [],
        "error": "",
    }

    with driver.session() as s:
        ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
        result["entity_count"] = ec
        if ec < 3:
            log.info("LTM graph too small (%d entities) — skipping GDS", ec)
            result["error"] = f"too_small:{ec}"
            return result

        edge_row = s.run("MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN count(r) as c").single()
        edge_count = edge_row["c"] if edge_row else 0
        result["edge_count"] = edge_count

        if not _gds_available(s):
            log.info("Neo4j GDS not installed — using Python-native pipeline")
            result = _run_python_native_pipeline(driver, result)
            elapsed_ms = (time.time() - t0) * 1000
            log.info(
                "Python-native GDS done: %d entities, %d communities, %.0fms",
                result["entity_count"],
                result["community_count"],
                elapsed_ms,
            )
            return result

    # ── Neo4j GDS plugin path ──────────────────────────────────────────────
    with driver.session() as s:
        try:
            _project_initial_graph(s, ec, edge_count)
            _run_node2vec(s)
            _run_cosine_weights(s)
            _project_undirected(s)
            _run_pagerank(s)
            _run_degree_centrality(s)
            _run_betweenness(s, ec)
            comm_result, algorithm_used = _run_community_detection(s)
            if comm_result:
                result["community_count"] = comm_result["communityCount"]
                result["modularity"] = comm_result["modularity"] or 0.0
                result["modularities"] = list(comm_result["modularities"] or [])
                result["ran_iterations"] = comm_result["ranLevels"]
                result["did_converge"] = True
                result["algorithm_used"] = algorithm_used
                try:
                    rows = s.run(
                        "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
                        "RETURN e.name as name, e.community_id as cid"
                    ).data()
                    members: dict[int, list[str]] = {}
                    for row in rows:
                        members.setdefault(row["cid"], []).append(row["name"])
                    result["changed_community_ids"] = list(members.keys())
                except Exception as exc:
                    log.warning("Community change detection failed: %s", exc)
            else:
                log.error("GDS community detection failed completely")
        finally:
            with contextlib.suppress(Exception):
                s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME).consume()

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        "GDS done: %d entities, %d communities, %.0fms",
        result["entity_count"],
        result["community_count"],
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Silhouette helpers (used by Neo4jStore._gds.py)
# ---------------------------------------------------------------------------


def _compute_silhouette(emb_rows: list[Any]) -> float | None:
    if len(emb_rows) < 10:
        return None
    x_emb = np.array([r["emb"] for r in emb_rows], dtype=np.float32)
    labels = np.array([r["cid"] for r in emb_rows])
    counts = Counter(labels)
    mask = np.array([counts[lbl] >= 2 for lbl in labels])
    x_filt = x_emb[mask]
    labels_filt = labels[mask]
    try:
        if len(np.unique(labels_filt)) >= 2 and len(x_filt) >= 4:
            return float(silhouette_score(x_filt, labels_filt, metric="cosine", random_state=42))
    except Exception as e_sil:
        log.warning("Silhouette score failed (non-fatal): %s", e_sil)
    return None


def _compute_separation(emb_rows: list[Any]) -> tuple[float | None, float | None]:
    counts = Counter(r["cid"] for r in emb_rows)
    comm_vecs: dict[Any, list[Any]] = defaultdict(list)
    for r in emb_rows:
        if counts[r["cid"]] >= 2:
            comm_vecs[r["cid"]].append(np.array(r["emb"], dtype=np.float32))
    if not comm_vecs:
        return None, None
    centroids = {cid: np.mean(vecs, axis=0) for cid, vecs in comm_vecs.items()}
    cid_list = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in cid_list])
    separated = total = 0
    for r in emb_rows:
        cid = r["cid"]
        if cid not in centroids:
            continue
        emb = np.array(r["emb"], dtype=np.float32)
        norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(emb)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = centroid_matrix @ emb / norms
        own_idx = cid_list.index(cid)
        if all(sims[own_idx] >= s_ for s_ in sims):
            separated += 1
        total += 1
    sep = separated / total if total > 0 else None
    return sep, (1.0 - sep if sep is not None else None)


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------


@dataclass
class FeatureEngineeringResult:
    entity_count: int = 0
    community_count: int = 0
    runtime_ms: float = 0.0
    top_entities: list[dict[str, Any]] = field(default_factory=list)
    algorithm_used: str = ""
    error: str = ""


class LTMEnrichmentService:
    """Stateless service for GDS enrichment and topic summarization."""

    def run_gds(self, driver: Any) -> GDSResult:
        raw = run_feature_engineering(driver)
        return GDSResult(
            entity_count=raw.get("entity_count", 0),
            edge_count=raw.get("edge_count", 0),
            community_count=raw.get("community_count", 0),
            modularity=raw.get("modularity", 0.0),
            modularities=raw.get("modularities", []),
            ran_iterations=raw.get("ran_iterations", 0),
            did_converge=raw.get("did_converge", False),
            changed_community_ids=raw.get("changed_community_ids", []),
            algorithm_used=raw.get("algorithm_used", ""),
            error=raw.get("error", ""),
        )

    def run_community_detection(self, driver: Any) -> CommunityResult:
        result = CommunityResult()
        with driver.session() as s:
            ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                result.error = f"too_small:{ec}"
                return result
        raw = run_feature_engineering(driver)
        result.community_count = raw.get("community_count", 0)
        result.modularity = raw.get("modularity", 0.0)
        result.modularities = raw.get("modularities", [])
        result.ran_iterations = raw.get("ran_iterations", 0)
        result.did_converge = raw.get("did_converge", False)
        result.algorithm_used = raw.get("algorithm_used", "")
        result.error = raw.get("error", "")
        return result

    def run_centrality(self, driver: Any) -> CentralityResult:
        result = CentralityResult()
        with driver.session() as s:
            ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                result.error = f"too_small:{ec}"
                return result
            edge_row = s.run("MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN count(r) as c").single()
            result.edge_count = edge_row["c"] if edge_row else 0
        return result

    async def run_topic_summarization(
        self,
        session_id: str,
        driver: Any,
        pg_store: Any,
        llm_key: str,
        llm_model: str,
        gds_result: Any | None = None,
    ) -> int:
        summarizer = None
        if llm_key:
            from graphknows.memory.ltm.topic_summarizer import TopicSummarizer

            summarizer = TopicSummarizer(llm_key, llm_model)
        return await driver.write_topic_summaries(
            session_id=session_id,
            summarizer=summarizer,
            pg_store=pg_store,
            gds_modularity=getattr(gds_result, "modularity", None),
            community_algorithm=getattr(gds_result, "algorithm_used", None),
        )

    async def run_gds_async(self, neo_store: Any) -> GDSResult:
        driver = getattr(neo_store, "_driver", neo_store)
        return await asyncio.to_thread(self.run_gds, driver)
