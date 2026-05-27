"""LTMEnrichmentService — single GDS enrichment API backed by feature_engineering helpers.

This module is the canonical home for GDS enrichment.
graphknows.feature_engineering is deprecated and delegates here.

# TODO(Wave 3): replace neo4j driver direct usage with IGraphDb adapter once #37 merges.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from graphknows.memory.models.gds import CentralityResult, CommunityResult, GDSResult

log = logging.getLogger("graphknows.memory.ltm.enrichment")

# ---------------------------------------------------------------------------
# Private helpers (migrated from graphknows.feature_engineering)
# ---------------------------------------------------------------------------

PROJECTED_GRAPH_NAME = "ltm-entities"
_DROP_GRAPH_CQL = "CALL gds.graph.drop($name, false) YIELD graphName"


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
                session_id="",
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
