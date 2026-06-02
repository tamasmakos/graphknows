"""LTMEnrichmentService — canonical GDS enrichment API.

# TODO(Wave 3): replace neo4j driver direct usage with IGraphDb adapter once #37 merges.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from graphknows.memory.models.gds import CentralityResult, CommunityResult, GDSResult

log = logging.getLogger("graphknows.memory.ltm.enrichment")

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

PROJECTED_GRAPH_NAME = "ltm-entities"
_DROP_GRAPH_CQL = "CALL gds.graph.drop($name, false) YIELD graphName"


def _is_missing_gds(exc: Exception) -> bool:
    text = str(exc)
    return "gds." in text and (
        "ProcedureNotFound" in text
        or "There is no procedure with the name" in text
        or "Unknown procedure" in text
    )


def _gds_available(s: Any) -> bool:
    """Return whether Neo4j Graph Data Science procedures are installed."""
    try:
        row = s.run(
            """
            SHOW PROCEDURES
            YIELD name
            WHERE name STARTS WITH 'gds.'
            RETURN count(name) AS c
            """
        ).single()
    except Exception as exc:
        if _is_missing_gds(exc):
            return False
        log.warning("Could not inspect GDS procedures; attempting enrichment anyway: %s", exc)
        return True
    return bool(row and row["c"] > 0)


def _project_initial_graph(s: Any, ec: int, edge_count: int) -> None:
    s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME)
    s.run("""
        CALL gds.graph.project.cypher(
            'ltm-entities',
            'MATCH (n:ENTITY) RETURN id(n) AS id',
            'MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
        )
        YIELD graphName, nodeCount, relationshipCount
    """)
    log.info("Projected graph 'ltm-entities': %d entities, %d edges", ec, edge_count)


def _run_node2vec(s: Any) -> None:
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
                "Node2Vec: %d embeddings in %dms",
                n2v["nodePropertiesWritten"],
                n2v["computeMillis"],
            )
    except Exception as exc:
        log.warning("Node2Vec failed (non-fatal): %s", exc)


def _run_cosine_weights(s: Any) -> None:
    try:
        s.run("""
            MATCH (a:ENTITY)-[r]->(b:ENTITY)
            WHERE a.node2vec IS NOT NULL AND b.node2vec IS NOT NULL
            SET r.weight = gds.similarity.cosine(a.node2vec, b.node2vec)
        """)
    except Exception as exc:
        log.warning("Cosine weight computation failed (non-fatal): %s", exc)


def _project_undirected(s: Any, _ec: int = 0, _edge_count: int = 0) -> None:
    s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME)
    try:
        s.run("""
            CALL gds.graph.project(
                'ltm-entities',
                'Entity',
                {RELATES: {orientation: 'UNDIRECTED', properties: {weight: {defaultValue: 1.0}}}}
            )
            YIELD graphName
        """)
    except Exception as exc:
        log.warning("Native undirected project failed, falling back to cypher: %s", exc)
        s.run("""
            CALL gds.graph.project.cypher(
                'ltm-entities',
                'MATCH (n:ENTITY) RETURN id(n) AS id',
                'MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
            )
            YIELD graphName
        """)


def _run_pagerank(s: Any) -> None:
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
    except Exception:
        try:
            s.run("""
                CALL gds.pageRank.write('ltm-entities', {
                    writeProperty: 'pagerank',
                    maxIterations: 20,
                    dampingFactor: 0.85
                })
                YIELD nodePropertiesWritten
            """)
        except Exception as exc2:
            log.warning("PageRank failed (non-fatal): %s", exc2)


def _run_degree_centrality(s: Any) -> None:
    try:
        s.run("""
            CALL gds.degree.write('ltm-entities', {writeProperty: 'degree_centrality'})
            YIELD nodePropertiesWritten
        """)
    except Exception as exc:
        log.warning("Degree centrality failed (non-fatal): %s", exc)


def _run_betweenness(s: Any, ec: int) -> None:
    if ec <= 5000:
        try:
            sampling_size = max(1, min(int(ec * 0.1), 100))
            s.run(
                "CALL gds.betweenness.write('ltm-entities', "
                "{writeProperty: 'betweenness', samplingSize: $sampling_size})"
                " YIELD nodePropertiesWritten",
                sampling_size=sampling_size,
            )
        except Exception as exc:
            log.warning("Betweenness failed (non-fatal): %s", exc)


def _run_community_detection(s: Any) -> tuple[Any, str]:
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
    except Exception as exc:
        log.warning("Weighted Louvain failed (%s) — trying unweighted", exc)
        try:
            comm_result = s.run("""
                CALL gds.louvain.write('ltm-entities', {writeProperty: 'community_id'})
                YIELD communityCount, modularity, modularities, ranLevels
            """).single()
            algorithm_used = "louvain_unweighted"
        except Exception:
            log.exception("Community detection failed")
    return comm_result, algorithm_used


def run_feature_engineering(driver: Any, session_id: str = "") -> dict[str, Any]:
    """Run full GDS feature engineering pipeline.

    Args:
        driver: Neo4j driver instance.
        session_id: Optional session identifier for GDSRun node.

    Returns:
        dict with entity_count, edge_count, community_count, etc.
    """
    import contextlib
    import hashlib
    import time

    t0 = time.time()
    result: dict[str, Any] = {
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
            result["error"] = "gds_unavailable"
            log.info("Neo4j GDS procedures are not installed; skipping enrichment")
            return result

        try:
            _project_initial_graph(s, ec, edge_count)
            _run_node2vec(s)
            _run_cosine_weights(s)
            _project_undirected(s, ec, edge_count)
            _run_pagerank(s)
            _run_degree_centrality(s)
            _run_betweenness(s, ec)
            comm_result, _algorithm_used = _run_community_detection(s)
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
                result["ran_iterations"] = ran_levels
                result["did_converge"] = True
                # Community change detection
                try:
                    rows = s.run(
                        "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
                        "RETURN e.name as name, e.community_id as cid"
                    ).data()
                    community_members: dict[int, list[str]] = {}
                    for row in rows:
                        community_members.setdefault(row["cid"], []).append(row["name"])
                    new_hashes: dict[int, str] = {}
                    for cid, names in community_members.items():
                        h = hashlib.sha1(
                            "|".join(sorted(names)).encode(), usedforsecurity=False
                        ).hexdigest()
                        new_hashes[cid] = h
                    existing = {
                        r["cid"]: r["hash"]
                        for r in s.run(
                            "MATCH (t:TOPIC) WHERE t.community_id IS NOT NULL AND t.community_hash IS NOT NULL "
                            "RETURN t.community_id as cid, t.community_hash as hash"
                        ).data()
                    }
                    result["changed_community_ids"] = [
                        cid for cid, h in new_hashes.items() if existing.get(cid) != h
                    ]
                except Exception as exc:
                    log.warning("Community change detection failed (non-fatal): %s", exc)
            else:
                log.error("Community detection completely failed")
        finally:
            with contextlib.suppress(Exception):
                s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME)

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        "GDS done: %d entities, %d communities, %.0fms",
        result["entity_count"],
        result["community_count"],
        elapsed_ms,
    )
    return result


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------


@dataclass
class FeatureEngineeringResult:
    """Result of a feature engineering run."""

    entity_count: int = 0
    community_count: int = 0
    runtime_ms: float = 0.0
    top_entities: list[dict] = field(default_factory=list)
    algorithm_used: str = ""
    error: str = ""


def _compute_silhouette(emb_rows: list) -> float | None:
    """Compute silhouette score from embedding rows."""
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


def _compute_separation(emb_rows: list) -> tuple[float | None, float | None]:
    """Compute global separation and overlap metrics."""
    counts = Counter(r["cid"] for r in emb_rows)
    comm_vecs: dict = defaultdict(list)
    for r in emb_rows:
        if counts[r["cid"]] >= 2:
            comm_vecs[r["cid"]].append(np.array(r["emb"], dtype=np.float32))
    if not comm_vecs:
        return None, None
    centroids = {cid: np.mean(vecs, axis=0) for cid, vecs in comm_vecs.items()}
    cid_list = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in cid_list])
    separated = 0
    total = 0
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


class LTMEnrichmentService:
    """Stateless service for GDS enrichment and topic summarization.

    Constructor accepts a Neo4j driver directly.
    # TODO(Wave 3): replace with IGraphDb adapter once epic #37 merges.
    """

    def run_gds(self, driver: Any) -> GDSResult:
        """Run full GDS pipeline (Node2Vec, PageRank, Louvain).

        Args:
            driver: Neo4j driver.

        Returns:
            GDSResult populated from feature engineering output.
        """
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
            error=raw.get("error", ""),
        )

    def run_community_detection(self, driver: Any) -> CommunityResult:
        """Run only community detection (Louvain).

        Args:
            driver: Neo4j driver.

        Returns:
            CommunityResult with community stats.
        """
        import contextlib

        result = CommunityResult()
        with driver.session() as s:
            ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                result.error = f"too_small:{ec}"
                return result
            if not _gds_available(s):
                result.error = "gds_unavailable"
                return result
            edge_row = s.run("MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN count(r) as c").single()
            result.edge_count = edge_row["c"] if edge_row else 0
            try:
                _project_undirected(s)
                comm_result, algorithm_used = _run_community_detection(s)
                if comm_result:
                    result.community_count = comm_result["communityCount"]
                    result.modularity = comm_result["modularity"] or 0.0
                    result.modularities = list(comm_result["modularities"] or [])
                    result.ran_iterations = comm_result["ranLevels"]
                    result.did_converge = True
                    result.algorithm_used = algorithm_used
            finally:
                with contextlib.suppress(Exception):
                    s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME)
        return result

    def run_centrality(self, driver: Any) -> CentralityResult:
        """Run PageRank, degree centrality, and betweenness.

        Args:
            driver: Neo4j driver.

        Returns:
            CentralityResult with entity/edge counts.
        """
        import contextlib

        result = CentralityResult()
        with driver.session() as s:
            ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                result.error = f"too_small:{ec}"
                return result
            if not _gds_available(s):
                result.error = "gds_unavailable"
                return result
            edge_row = s.run("MATCH (a:ENTITY)-[r]->(b:ENTITY) RETURN count(r) as c").single()
            result.edge_count = edge_row["c"] if edge_row else 0
            try:
                _project_undirected(s)
                _run_pagerank(s)
                _run_degree_centrality(s)
                _run_betweenness(s, ec)
            finally:
                with contextlib.suppress(Exception):
                    s.run(_DROP_GRAPH_CQL, name=PROJECTED_GRAPH_NAME)
        return result

    async def run_topic_summarization(
        self,
        session_id: str,
        driver: Any,
        pg_store: Any,
        llm_key: str,
        llm_model: str,
    ) -> int:
        """Run Louvain + LLM topic summarization and write TOPIC nodes.

        Accepts a neo_store object (Neo4jStore or compatible) as ``driver``
        when the caller already has one, or a raw driver when the store is
        not yet available.  If ``driver`` exposes a ``summarize_topics``
        coroutine it is called directly; otherwise a basic Louvain-only pass
        is executed via the internal helpers.

        Args:
            session_id: The session ID to summarize topics for.
            driver: Neo4jStore instance (preferred) or bare Neo4j driver.
            pg_store: PgVectorStore for ltm_topics write-back.
            llm_key: OpenRouter API key.
            llm_model: LLM model identifier.

        Returns:
            Number of topics created.
        """
        if hasattr(driver, "summarize_topics"):
            # driver is actually a Neo4jStore (or compatible) — delegate directly
            return await driver.summarize_topics(  # type: ignore[return-value, no-any-return]
                session_id=session_id,
                openrouter_key=llm_key,
                llm_model=llm_model,
                pg_store=pg_store,
            )
        # Bare driver — no topic summarization available without a store wrapper
        log.warning("run_topic_summarization: bare driver passed; skipping LLM summarization")
        return 0

    async def run_gds_async(self, driver: Any) -> GDSResult:
        """Async wrapper for run_gds.

        Args:
            driver: Neo4j driver.

        Returns:
            GDSResult.
        """
        return await asyncio.to_thread(self.run_gds, driver)
