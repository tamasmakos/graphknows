"""LTM Feature Engineering — Node2Vec + weighted Leiden community detection + centrality measures."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

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

        edge_count_row = s.run(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN count(r) as c"
        ).single()
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

            # ── Step 4: Re-project with weights for weighted Leiden ──
            s.run("CALL gds.graph.drop($name, false) YIELD graphName", name=PROJECTED_GRAPH_NAME)
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
            except Exception as e:
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

            # ── Step 8: Weighted Leiden community detection ──
            leiden_result = None
            try:
                leiden_result = s.run("""
                    CALL gds.leiden.write('ltm-entities', {
                        writeProperty: 'community_id',
                        relationshipWeightProperty: 'weight',
                        includeIntermediateCommunities: false,
                        gamma: 1.0,
                        theta: 0.01
                    })
                    YIELD communityCount, modularity, modularities, ranIterations, didConverge
                """).single()
            except Exception as e:
                log.warning("Weighted Leiden failed (%s) — trying unweighted", e)
                try:
                    leiden_result = s.run("""
                        CALL gds.leiden.write('ltm-entities', {
                            writeProperty: 'community_id',
                            gamma: 1.0,
                            theta: 0.01
                        })
                        YIELD communityCount, modularity, modularities, ranIterations, didConverge
                    """).single()
                except Exception as e2:
                    log.error("Leiden failed: %s", e2)

            if leiden_result:
                community_count = leiden_result["communityCount"]
                modularity = leiden_result["modularity"] or 0.0
                modularities = list(leiden_result["modularities"]) if leiden_result["modularities"] else []
                ran_iterations = leiden_result["ranIterations"]
                did_converge = leiden_result["didConverge"]

                result["community_count"] = community_count
                result["modularity"] = modularity
                result["modularities"] = modularities
                result["ran_iterations"] = ran_iterations
                result["did_converge"] = did_converge

                log.info(
                    "Leiden: %d communities, modularity=%.4f, iterations=%d, converged=%s",
                    community_count, modularity, ran_iterations, did_converge,
                )

                # ── Step 9: Store GDSRun node ──
                if session_id:
                    try:
                        s.run(
                            "MERGE (r:GDSRun {session_id: $session_id}) "
                            "SET r.ran_at = datetime(), "
                            "    r.community_count = $community_count, "
                            "    r.modularity = $modularity, "
                            "    r.modularities = $modularities, "
                            "    r.ran_iterations = $ran_iterations, "
                            "    r.did_converge = $did_converge, "
                            "    r.entity_count = $entity_count, "
                            "    r.edge_count = $edge_count",
                            session_id=session_id,
                            community_count=community_count,
                            modularity=modularity,
                            modularities=modularities,
                            ran_iterations=ran_iterations,
                            did_converge=did_converge,
                            entity_count=ec,
                            edge_count=edge_count,
                        )
                    except Exception as e:
                        log.warning("GDSRun node creation failed (non-fatal): %s", e)

                # ── Step 10: Community change detection ──
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
                        h = hashlib.sha1("|".join(sorted_names).encode()).hexdigest()
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
                # Leiden failed — fall back to Louvain
                try:
                    s.run("""
                        CALL gds.louvain.write('ltm-entities', {writeProperty: 'community_id'})
                        YIELD communityCount, modularity
                    """)
                    cc = s.run(
                        "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                        "RETURN count(DISTINCT e.community_id) as c"
                    ).single()["c"]
                    result["community_count"] = cc
                    log.info("Louvain fallback: %d communities", cc)
                except Exception as e2:
                    log.error("Community detection completely failed: %s", e2)
                    result["error"] = f"community_failed:{e2}"

        finally:
            try:
                s.run("CALL gds.graph.drop($name, false) YIELD graphName", name=PROJECTED_GRAPH_NAME)
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
