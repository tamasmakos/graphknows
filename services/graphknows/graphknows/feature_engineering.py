"""LTM Feature Engineering — Node2Vec + weighted Leiden community detection + centrality measures."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

log = logging.getLogger("graphknows.feature_engineering")

PROJECTED_GRAPH_NAME = "ltm-entities"


@dataclass
class FeatureEngineeringResult:
    entity_count: int = 0
    community_count: int = 0
    runtime_ms: float = 0.0
    top_entities: list[dict] = field(default_factory=list)
    algorithm_used: str = ""
    error: str = ""


class LTMFeatureEngineer:
    """Runs GDS algorithms on Neo4j LTM entity graph after each session write-back.

    Writes pagerank, betweenness, degree_centrality, community_id directly to Entity nodes.
    """

    def __init__(self, neo4j_driver: Any) -> None:
        self._driver = neo4j_driver

    def run(self) -> FeatureEngineeringResult:
        """Run full feature engineering pipeline synchronously (legacy interface).

        Returns:
            FeatureEngineeringResult with counts and top entities by PageRank.
        """
        result_dict = run_feature_engineering(self._driver)
        result = FeatureEngineeringResult()
        result.entity_count = result_dict.get("entity_count", 0)
        result.community_count = result_dict.get("community_count", 0)
        result.algorithm_used = "leiden"
        result.error = result_dict.get("error", "")
        return result


def run_feature_engineering(driver_or_session_id: Any, session_id: str = "") -> dict:
    """Run full GDS feature engineering pipeline.

    Can be called as:
        run_feature_engineering(neo4j_driver)
        run_feature_engineering(neo4j_driver, session_id="abc")

    Args:
        driver_or_session_id: neo4j driver instance
        session_id: optional session identifier for GDSRun node

    Returns:
        dict with entity_count, edge_count, community_count, modularity,
        modularities, ran_iterations, did_converge, changed_community_ids
    """
    driver = driver_or_session_id
    t0 = time.time()

    result: dict = {
        "entity_count": 0,
        "edge_count": 0,
        "community_count": 0,
        "modularity": 0.0,
        "modularities": [],
        "ran_iterations": 0,
        "did_converge": False,
        "changed_community_ids": [],
        "error": "",
    }

    with driver.session() as s:
        ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
        result["entity_count"] = ec
        if ec < 3:
            log.info("LTM graph too small (%d entities) — skipping feature engineering", ec)
            result["error"] = f"too_small:{ec}"
            return result

        edge_count_row = s.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN count(r) as c").single()
        edge_count = edge_count_row["c"] if edge_count_row else 0
        result["edge_count"] = edge_count

        try:
            # ── Step 1: Drop old projection and project ALL Entity→Entity edges ──
            s.run("CALL gds.graph.drop($name, false) YIELD graphName", name=PROJECTED_GRAPH_NAME)

            s.run("""
                CALL gds.graph.project.cypher(
                    'ltm-entities',
                    'MATCH (n:Entity) RETURN id(n) AS id',
                    'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
                )
                YIELD graphName, nodeCount, relationshipCount
            """)
            log.info("Projected graph 'ltm-entities': %d entities, %d edges", ec, edge_count)

            # ── Step 2: Run Node2Vec to produce embeddings ──
            try:
                n2v = s.run("""
                    CALL gds.node2vec.write('ltm-entities', {
                        writeProperty: 'node2vec',
                        embeddingDimension: 64,
                        walkLength: 30,
                        walksPerNode: 10,
                        inOutFactor: 1.0,
                        returnFactor: 1.0,
                        iterations: 5
                    })
                    YIELD nodePropertiesWritten, computeMillis
                """).single()
                if n2v:
                    log.info(
                        "Node2Vec: %d embeddings written in %dms",
                        n2v["nodePropertiesWritten"],
                        n2v["computeMillis"],
                    )
            except Exception as e:
                log.warning("Node2Vec failed (non-fatal): %s", e)

            # ── Step 3: Compute cosine similarity weights on all entity-entity edges ──
            try:
                s.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.node2vec IS NOT NULL AND b.node2vec IS NOT NULL
                    SET r.weight = gds.similarity.cosine(a.node2vec, b.node2vec)
                """)
                log.info("Cosine similarity weights written to entity-entity edges")
            except Exception as e:
                log.warning("Cosine weight computation failed (non-fatal): %s", e)

            # ── Step 4: Re-project as UNDIRECTED with weights for weighted Leiden ──
            s.run("CALL gds.graph.drop($name, false) YIELD graphName", name=PROJECTED_GRAPH_NAME)
            try:
                s.run("""
                    CALL gds.graph.project(
                        'ltm-entities',
                        'Entity',
                        {RELATES: {orientation: 'UNDIRECTED', properties: {weight: {defaultValue: 1.0}}}}
                    )
                    YIELD graphName
                """)
            except Exception as e:
                log.warning("Native undirected project failed, falling back to cypher: %s", e)
                s.run("""
                    CALL gds.graph.project.cypher(
                        'ltm-entities',
                        'MATCH (n:Entity) RETURN id(n) AS id',
                        'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
                    )
                    YIELD graphName
                """)

            # ── Step 5: PageRank ──
            try:
                s.run("""
                    CALL gds.pageRank.write('ltm-entities', {
                        writeProperty: 'pagerank',
                        maxIterations: 20,
                        dampingFactor: 0.85,
                        relationshipWeightProperty: 'weight'
                    })
                    YIELD nodePropertiesWritten
                """)
                log.info("PageRank written")
            except Exception:
                # Fallback without weight property
                try:
                    s.run("""
                        CALL gds.pageRank.write('ltm-entities', {
                            writeProperty: 'pagerank',
                            maxIterations: 20,
                            dampingFactor: 0.85
                        })
                        YIELD nodePropertiesWritten
                    """)
                    log.info("PageRank written (unweighted fallback)")
                except Exception as e2:
                    log.warning("PageRank failed (non-fatal): %s", e2)

            # ── Step 6: Degree centrality ──
            try:
                s.run("""
                    CALL gds.degree.write('ltm-entities', {writeProperty: 'degree_centrality'})
                    YIELD nodePropertiesWritten
                """)
            except Exception as e:
                log.warning("Degree centrality failed (non-fatal): %s", e)

            # ── Step 7: Betweenness (sampling for large graphs) ──
            if ec <= 5000:
                try:
                    sampling_size = max(1, min(int(ec * 0.1), 100))
                    s.run(
                        "CALL gds.betweenness.write('ltm-entities', "
                        "{writeProperty: 'betweenness', samplingSize: $sampling_size})"
                        " YIELD nodePropertiesWritten",
                        sampling_size=sampling_size,
                    )
                except Exception as e:
                    log.warning("Betweenness failed (non-fatal): %s", e)

            # ── Step 8: Weighted Louvain community detection ──
            # GDS 2026.04 — gds.leiden.write silently returns no rows; use Louvain.
            # Louvain YIELD: communityCount, modularity, modularities, ranLevels
            comm_result = None
            algorithm_used = "louvain"
            try:
                comm_result = s.run("""
                    CALL gds.louvain.write('ltm-entities', {
                        writeProperty: 'community_id',
                        relationshipWeightProperty: 'weight'
                    })
                    YIELD communityCount, modularity, modularities, ranLevels
                """).single()
            except Exception as e:
                log.warning("Weighted Louvain failed (%s) — trying unweighted", e)
                try:
                    comm_result = s.run("""
                        CALL gds.louvain.write('ltm-entities', {writeProperty: 'community_id'})
                        YIELD communityCount, modularity, modularities, ranLevels
                    """).single()
                    algorithm_used = "louvain_unweighted"
                except Exception as e2:
                    log.error("Community detection failed: %s", e2)

            if comm_result:
                community_count = comm_result["communityCount"]
                modularity = comm_result["modularity"] or 0.0
                modularities = (
                    list(comm_result["modularities"]) if comm_result["modularities"] else []
                )
                ran_levels = comm_result["ranLevels"]

                result["community_count"] = community_count
                result["modularity"] = modularity
                result["modularities"] = modularities
                result["ran_iterations"] = ran_levels  # ranLevels maps to ran_iterations field
                result["did_converge"] = True  # Louvain always converges

                log.info(
                    "%s: %d communities, modularity=%.4f, levels=%d",
                    algorithm_used,
                    community_count,
                    modularity,
                    ran_levels,
                )

                # ── Step 9: Compute quality metrics ──
                run_id = session_id or f"global-{int(time.time())}"
                sil = None
                global_separation = None
                global_overlap = None
                unweighted_modularity = None

                try:
                    emb_rows = s.run(
                        "MATCH (e:Entity) WHERE e.node2vec IS NOT NULL AND e.community_id IS NOT NULL "
                        "RETURN e.node2vec as emb, e.community_id as cid"
                    ).data()

                    if len(emb_rows) >= 10:
                        X = np.array([r["emb"] for r in emb_rows], dtype=np.float32)
                        labels = np.array([r["cid"] for r in emb_rows])
                        counts = Counter(labels)
                        mask = np.array([counts[l] >= 2 for l in labels])
                        X_filt = X[mask]
                        labels_filt = labels[mask]
                        try:
                            if len(np.unique(labels_filt)) >= 2 and len(X_filt) >= 4:
                                sil = float(silhouette_score(X_filt, labels_filt, metric="cosine"))
                        except Exception as e_sil:
                            log.warning("Silhouette score failed (non-fatal): %s", e_sil)

                        comm_vecs: dict = defaultdict(list)
                        for r in emb_rows:
                            if counts[r["cid"]] >= 2:
                                comm_vecs[r["cid"]].append(np.array(r["emb"], dtype=np.float32))

                        if comm_vecs:
                            centroids = {
                                cid: np.mean(vecs, axis=0) for cid, vecs in comm_vecs.items()
                            }
                            cid_list = list(centroids.keys())
                            centroid_matrix = np.stack([centroids[c] for c in cid_list])

                            separated = 0
                            total = 0
                            for r in emb_rows:
                                cid = r["cid"]
                                if cid not in centroids:
                                    continue
                                emb = np.array(r["emb"], dtype=np.float32)
                                norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(
                                    emb
                                )
                                norms = np.where(norms == 0, 1e-9, norms)
                                sims = centroid_matrix @ emb / norms
                                own_idx = cid_list.index(cid)
                                own_sim = sims[own_idx]
                                if all(own_sim >= s_ for s_ in sims):
                                    separated += 1
                                total += 1

                            global_separation = separated / total if total > 0 else None
                            global_overlap = (
                                1.0 - global_separation if global_separation is not None else None
                            )
                except Exception as e_qm:
                    log.warning("Quality metric computation failed (non-fatal): %s", e_qm)

                try:
                    unweighted_row = s.run("""
                        CALL gds.louvain.stats('ltm-entities')
                        YIELD modularity
                    """).single()
                    unweighted_modularity = (
                        float(unweighted_row["modularity"]) if unweighted_row else None
                    )
                except Exception:
                    try:
                        unweighted_row = s.run("""
                            CALL gds.louvain.stats('ltm-entities', {gamma: 1.0})
                            YIELD modularity
                        """).single()
                        unweighted_modularity = (
                            float(unweighted_row["modularity"]) if unweighted_row else None
                        )
                    except Exception:
                        unweighted_modularity = None

                result["silhouette_cosine"] = sil
                result["global_separation"] = global_separation
                result["global_overlap"] = global_overlap
                result["unweighted_modularity"] = unweighted_modularity

                # ── Step 10: Store GDSRun node ──
                try:
                    s.run(
                        "MERGE (r:GDSRun {session_id: $session_id}) "
                        "SET r.ran_at = datetime(), "
                        "    r.community_count = $community_count, "
                        "    r.modularity = $modularity, "
                        "    r.unweighted_modularity = $unweighted_modularity, "
                        "    r.modularities = $modularities, "
                        "    r.ran_iterations = $ran_iterations, "
                        "    r.did_converge = $did_converge, "
                        "    r.entity_count = $entity_count, "
                        "    r.edge_count = $edge_count, "
                        "    r.silhouette_cosine = $sil, "
                        "    r.global_separation = $gsep, "
                        "    r.global_overlap = $goverlap",
                        session_id=run_id,
                        community_count=community_count,
                        modularity=modularity,
                        unweighted_modularity=unweighted_modularity,
                        modularities=modularities,
                        ran_iterations=ran_levels,
                        did_converge=True,
                        entity_count=ec,
                        edge_count=edge_count,
                        sil=sil,
                        gsep=global_separation,
                        goverlap=global_overlap,
                    )
                except Exception as e:
                    log.warning("GDSRun node creation failed (non-fatal): %s", e)

                # ── Step 11: Community change detection ──
                try:
                    rows = s.run(
                        "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                        "RETURN e.name as name, e.community_id as cid"
                    ).data()

                    # Group entities by community_id
                    community_members: dict[int, list[str]] = {}
                    for row in rows:
                        cid = row["cid"]
                        name = row["name"]
                        community_members.setdefault(cid, []).append(name)

                    # Compute SHA1 hash for each community (sorted entity names)
                    new_hashes: dict[int, str] = {}
                    for cid, names in community_members.items():
                        sorted_names = sorted(names)
                        h = hashlib.sha1(
                            "|".join(sorted_names).encode(), usedforsecurity=False
                        ).hexdigest()
                        new_hashes[cid] = h

                    # Fetch existing hashes from Topic nodes
                    existing_hash_rows = s.run(
                        "MATCH (t:Topic) WHERE t.community_id IS NOT NULL AND t.community_hash IS NOT NULL "
                        "RETURN t.community_id as cid, t.community_hash as hash"
                    ).data()
                    existing_hashes = {r["cid"]: r["hash"] for r in existing_hash_rows}

                    changed_community_ids: list[int] = []
                    for cid, new_hash in new_hashes.items():
                        if existing_hashes.get(cid) != new_hash:
                            changed_community_ids.append(cid)

                    result["changed_community_ids"] = changed_community_ids
                    log.info(
                        "Community change detection: %d changed out of %d total",
                        len(changed_community_ids),
                        len(new_hashes),
                    )
                except Exception as e:
                    log.warning("Community change detection failed (non-fatal): %s", e)

            else:
                # All community detection failed — log only
                log.error("Community detection completely failed")

        finally:
            try:
                s.run(
                    "CALL gds.graph.drop($name, false) YIELD graphName", name=PROJECTED_GRAPH_NAME
                )
            except Exception:
                pass

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        "Feature engineering done: %d entities, %d edges, %d communities, %.0fms",
        result["entity_count"],
        result["edge_count"],
        result["community_count"],
        elapsed_ms,
    )
    return result
