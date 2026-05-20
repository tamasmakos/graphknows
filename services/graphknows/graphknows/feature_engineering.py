"""LTM Feature Engineering — Leiden community detection + centrality measures."""

from __future__ import annotations

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
        """Run full feature engineering pipeline synchronously.

        Returns:
            FeatureEngineeringResult with counts and top entities by PageRank.
        """
        t0 = time.time()
        result = FeatureEngineeringResult()

        with self._driver.session() as s:
            ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                log.info("LTM graph too small (%d entities) — skipping feature engineering", ec)
                result.error = f"too_small:{ec}"
                return result

            try:
                try:
                    s.run(f"CALL gds.graph.drop('{PROJECTED_GRAPH_NAME}', false)")
                except Exception:
                    pass

                s.run(
                    f"CALL gds.graph.project('{PROJECTED_GRAPH_NAME}', 'Entity', "
                    f"{{RELATES: {{orientation: 'UNDIRECTED'}}}})"
                )

                s.run(
                    f"CALL gds.pageRank.write('{PROJECTED_GRAPH_NAME}', "
                    f"{{writeProperty: 'pagerank', maxIterations: 20, dampingFactor: 0.85}})"
                )

                s.run(
                    f"CALL gds.degree.write('{PROJECTED_GRAPH_NAME}', "
                    f"{{writeProperty: 'degree_centrality'}})"
                )

                if ec <= 5000:
                    try:
                        sampling_size = max(1, min(int(ec * 0.1), 100))
                        s.run(
                            f"CALL gds.betweenness.write('{PROJECTED_GRAPH_NAME}', "
                            f"{{writeProperty: 'betweenness', samplingSize: $sampling_size}})",
                            sampling_size=sampling_size,
                        )
                    except Exception as e:
                        log.warning("betweenness failed (non-fatal): %s", e)

                community_algo = "leiden"
                try:
                    s.run(
                        f"CALL gds.leiden.write('{PROJECTED_GRAPH_NAME}', "
                        f"{{writeProperty: 'community_id', gamma: 1.0, maxLevels: 10}})"
                    )
                    result.algorithm_used = "leiden"
                except Exception as e:
                    log.warning("Leiden unavailable (%s) — falling back to Louvain", e)
                    try:
                        s.run(
                            f"CALL gds.louvain.write('{PROJECTED_GRAPH_NAME}', "
                            f"{{writeProperty: 'community_id'}})"
                        )
                        result.algorithm_used = "louvain"
                    except Exception as e2:
                        log.error("Community detection failed: %s", e2)
                        result.error = f"community_failed:{e2}"

                cc = s.run(
                    "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                    "RETURN count(DISTINCT e.community_id) as c"
                ).single()["c"]
                result.community_count = cc

                top = s.run(
                    "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL "
                    "RETURN e.name as name, e.type as type, e.pagerank as pr, "
                    "e.community_id as community "
                    "ORDER BY e.pagerank DESC LIMIT 10"
                ).data()
                result.top_entities = top

            finally:
                try:
                    s.run(f"CALL gds.graph.drop('{PROJECTED_GRAPH_NAME}', false)")
                except Exception:
                    pass

        result.runtime_ms = (time.time() - t0) * 1000
        log.info(
            "Feature engineering done: %d entities, %d communities, algo=%s, %.0fms",
            result.entity_count, result.community_count, result.algorithm_used, result.runtime_ms,
        )
        return result
