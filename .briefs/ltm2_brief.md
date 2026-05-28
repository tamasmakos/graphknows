# LTM-2 Brief: Feature Engineering — Leiden Clustering + Centrality on LTM Entity Graph

## Working Directory
/root/graphknows

## Your Tools (native DB access via .opencode/tools/)
- neo4j-query.ts  → Cypher against Neo4j bolt://172.18.0.1:7687
- memgraph-query.ts → Cypher against Memgraph bolt://172.18.0.1:17687 (no auth)
- pgvector-query.ts → SQL against PostgreSQL 172.18.0.1:5432

## Model
claude-sonnet-4.6 via GitHub Copilot

## Context: What Exists
- LTM-1 established the Neo4j schema: Entity, Segment, Topic nodes with constraints
- Entity nodes now have: name, type, first_seen, last_seen, confidence_avg
- We need to add: pagerank, betweenness, degree_centrality, community_id

## Step 1: Check what GDS algorithms are available
Use neo4j-query.ts to run:
```cypher
CALL gds.list() YIELD name, description
WHERE name CONTAINS 'pageRank' OR name CONTAINS 'betweenness' OR name CONTAINS 'leiden' OR name CONTAINS 'louvain' OR name CONTAINS 'degree'
RETURN name, description
```
Also check entity count:
```cypher
MATCH (e:Entity) RETURN count(e) as entity_count
MATCH ()-[r:RELATES]->() RETURN count(r) as rel_count
```

## Step 2: Create services/graphknows/feature_engineering.py

```python
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
            # Check entity count — skip if graph too small
            ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
            result.entity_count = ec
            if ec < 3:
                log.info("LTM graph too small (%d entities) — skipping feature engineering", ec)
                result.error = f"too_small:{ec}"
                return result

            try:
                # Drop projected graph if stale
                try:
                    s.run(f"CALL gds.graph.drop('{PROJECTED_GRAPH_NAME}', false)")
                except Exception:
                    pass

                # Project entity graph (undirected for community detection)
                s.run(
                    f"CALL gds.graph.project('{PROJECTED_GRAPH_NAME}', 'Entity', "
                    "{{RELATES: {{orientation: 'UNDIRECTED'}}}})"
                )

                # PageRank
                s.run(
                    f"CALL gds.pageRank.write('{PROJECTED_GRAPH_NAME}', "
                    "{{writeProperty: 'pagerank', maxIterations: 20, dampingFactor: 0.85}})"
                )

                # Degree centrality
                s.run(
                    f"CALL gds.degree.write('{PROJECTED_GRAPH_NAME}', "
                    "{{writeProperty: 'degree_centrality'}})"
                )

                # Betweenness — skip on large graphs (O(n*m))
                if ec <= 5000:
                    try:
                        s.run(
                            f"CALL gds.betweenness.write('{PROJECTED_GRAPH_NAME}', "
                            "{{writeProperty: 'betweenness', samplingSize: min(toInteger($n*0.1), 100)}})",
                            n=ec,
                        )
                    except Exception as e:
                        log.warning("betweenness failed (non-fatal): %s", e)

                # Community detection: try Leiden first, fall back to Louvain
                community_algo = "leiden"
                try:
                    s.run(
                        f"CALL gds.leiden.write('{PROJECTED_GRAPH_NAME}', "
                        "{{writeProperty: 'community_id', gamma: 1.0, maxLevels: 10}})"
                    )
                    result.algorithm_used = "leiden"
                except Exception as e:
                    log.warning("Leiden unavailable (%s) — falling back to Louvain", e)
                    try:
                        s.run(
                            f"CALL gds.louvain.write('{PROJECTED_GRAPH_NAME}', "
                            "{{writeProperty: 'community_id'}})"
                        )
                        result.algorithm_used = "louvain"
                    except Exception as e2:
                        log.error("Community detection failed: %s", e2)
                        result.error = f"community_failed:{e2}"

                # Count communities
                cc = s.run(
                    "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                    "RETURN count(DISTINCT e.community_id) as c"
                ).single()["c"]
                result.community_count = cc

                # Top 10 entities by pagerank
                top = s.run(
                    "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL "
                    "RETURN e.name as name, e.type as type, e.pagerank as pr, "
                    "e.community_id as community "
                    "ORDER BY e.pagerank DESC LIMIT 10"
                ).data()
                result.top_entities = top

            finally:
                # Always drop projected graph
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
```

## Step 3: Wire into WriteBackService

In services/agent/session/writeback.py, at the end of write_back(), after the Neo4j writes,
add a call to LTMFeatureEngineer:

```python
# After all writes complete (still inside the method):
from graphknows.feature_engineering import LTMFeatureEngineer

try:
    from neo4j import GraphDatabase as _GDB
    sync_driver = _GDB.driver(self._uri, auth=(self._user, self._password))
    fe = LTMFeatureEngineer(sync_driver)
    fe_result = fe.run()
    log.info("Feature engineering: %s", fe_result)
    sync_driver.close()
except Exception:
    log.exception("Feature engineering failed — non-fatal")
```

## Step 4: Verify

Use neo4j-query.ts to run after implementation:
```cypher
MATCH (e:Entity)
WHERE e.pagerank IS NOT NULL
RETURN e.name, e.type, e.pagerank, e.community_id, e.betweenness
ORDER BY e.pagerank DESC LIMIT 10
```

Also verify communities formed:
```cypher
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(e) as size, collect(e.name)[0..3] as sample_names
ORDER BY size DESC LIMIT 10
```

Also import check:
```bash
PYTHONPATH=/root/graphknows/services:/root/graphknows/services/graphknows python3 -B -c   "from graphknows.feature_engineering import LTMFeatureEngineer; print('OK')"
```

## Acceptance Criteria
- services/graphknows/feature_engineering.py created and importable
- LTMFeatureEngineer.run() works on live Neo4j (even if small/empty — graceful skip)
- Entity nodes get pagerank, community_id written (verify with neo4j-query.ts)
- GDS unavailability is caught gracefully with a log warning, not a crash
- WriteBackService calls feature engineering after each write-back
- Do NOT commit — leave for principal review

## Pitfalls
- GDS projected graph MUST be dropped in a finally block — stale projections block re-runs
- Leiden is GDS Enterprise; always fall back to Louvain if it raises
- betweenness is expensive — guard with entity_count <= 5000
- services/graphknows/ may have its own package structure — check __init__.py exists
- Import path for graphknows package: PYTHONPATH must include /root/graphknows/services/graphknows
