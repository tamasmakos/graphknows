# Brief: LTM-6 — Feature Engineering Rewrite

## Context
File: `/root/graphknows/services/graphknows/graphknows/feature_engineering.py`
Topic summarizer: `/root/graphknows/services/graphknows/graphknows/topic_summarizer.py`
Neo4j URI from env: `NEO4J_URI`, user: `NEO4J_USER`, password: `NEO4J_PASSWORD`
Load dotenv from `/root/graphknows/.env`

Neo4j GDS 2.x is available. All these procedures exist and are confirmed working:
- `gds.node2vec.write` / `gds.node2vec.stream`
- `gds.leiden.write` — YIELDS `communityCount, modularity, modularities, ranIterations, didConverge`
- `gds.similarity.cosine(list1, list2)` — scalar function
- `gds.pageRank.write`, `gds.betweenness.write`, `gds.degree.write`

## Current Bug
The GDS projection uses `{RELATES: {orientation: 'UNDIRECTED'}}` which only captures 18 edges.
The real graph has 1355 Entity→Entity edges:
- CO_OCCURS: 1288
- RELATION: 49
- RELATES: 18

GDS sees only 18 edges → ~375 singleton communities → modularity near 0.

## Required Changes to `feature_engineering.py`

### 1. Fix the GDS projection to include ALL Entity→Entity edges
Use Cypher projection (not native projection) because native projection cannot use wildcards:

```python
# Drop existing projected graph if it exists
session.run("CALL gds.graph.drop('ltm-entities', false) YIELD graphName")

# Project ALL entity-entity edges of any type
session.run("""
    CALL gds.graph.project.cypher(
        'ltm-entities',
        'MATCH (n:Entity) RETURN id(n) AS id',
        'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
    )
    YIELD graphName, nodeCount, relationshipCount
""")
```

Note: Use `gds.graph.project.cypher` not `gds.graph.project` for cypher-based projections in GDS 2.x.

### 2. Run Node2Vec on the projected graph
```python
result = session.run("""
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
logger.info(f"Node2Vec: {result['nodePropertiesWritten']} embeddings written in {result['computeMillis']}ms")
```

### 3. Compute cosine similarity weights on all entity-entity edges and persist as r.weight
```python
session.run("""
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.node2vec IS NOT NULL AND b.node2vec IS NOT NULL
    SET r.weight = gds.similarity.cosine(a.node2vec, b.node2vec)
""")
```

### 4. Drop and re-project with the new weights, then run weighted Leiden
```python
session.run("CALL gds.graph.drop('ltm-entities', false) YIELD graphName")
session.run("""
    CALL gds.graph.project.cypher(
        'ltm-entities',
        'MATCH (n:Entity) RETURN id(n) AS id',
        'MATCH (a:Entity)-[r]->(b:Entity) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
    )
    YIELD graphName
""")

leiden_result = session.run("""
    CALL gds.leiden.write('ltm-entities', {
        writeProperty: 'community_id',
        relationshipWeightProperty: 'weight',
        includeIntermediateCommunities: false,
        gamma: 1.0,
        theta: 0.01
    })
    YIELD communityCount, modularity, modularities, ranIterations, didConverge
""").single()

logger.info(f"Leiden: {leiden_result['communityCount']} communities, modularity={leiden_result['modularity']:.4f}, iterations={leiden_result['ranIterations']}, converged={leiden_result['didConverge']}")
```

### 5. Store modularity in a GDSRun node in Neo4j
```python
session.run("""
    MERGE (r:GDSRun {session_id: $session_id})
    SET r.ran_at = datetime(),
        r.community_count = $community_count,
        r.modularity = $modularity,
        r.modularities = $modularities,
        r.ran_iterations = $ran_iterations,
        r.did_converge = $did_converge,
        r.entity_count = $entity_count,
        r.edge_count = $edge_count
""", session_id=session_id, community_count=leiden_result['communityCount'],
    modularity=leiden_result['modularity'],
    modularities=list(leiden_result['modularities']) if leiden_result['modularities'] else [],
    ran_iterations=leiden_result['ranIterations'],
    did_converge=leiden_result['didConverge'],
    entity_count=..., edge_count=...)
```

### 6. Run PageRank and betweenness on the same weighted graph

Keep existing PageRank logic but make sure it targets the new projection. Keep the betweenness write too.

### 7. Community change detection
After writing community_id to all entities, compute a hash of (entity_name → community_id) mapping per community and compare with the stored hash on existing Topic nodes. Return a list of `changed_community_ids` — these are community_ids where the set of members changed (new entities added, entities moved, or community is brand new).

Store the hash on the Topic node as `community_hash: str`.

Return the full leiden stats AND the `changed_community_ids` from `run_feature_engineering(session_id)`.

## Required Changes to `topic_summarizer.py`

### 1. Write entity_count on Topic creation/update
```python
# After MERGE Topic node:
session.run("""
    MATCH (t:Topic {community_id: $cid})
    MATCH (e:Entity {community_id: $cid})
    WITH t, count(e) AS cnt
    SET t.entity_count = cnt
""", cid=community_id)
```

### 2. Accept changed_community_ids parameter
```python
async def summarize_topics(self, changed_community_ids: list[int] | None = None) -> int:
```
If `changed_community_ids` is None → process all communities.
If provided → only re-summarize topics whose `community_id` is in `changed_community_ids`.

### 3. Two-level hierarchy: Topics then SubTopics
After writing Topic nodes, for each topic community with entity_count >= 6, run a second Leiden pass on just those entities to find sub-communities:

```python
# For community with id=X having >= 6 entities
# Drop old sub-projection if exists
session.run("CALL gds.graph.drop('ltm-subtopic-{cid}', false) YIELD graphName")
session.run(f"""
    CALL gds.graph.project.cypher(
        'ltm-subtopic-{cid}',
        'MATCH (n:Entity {{community_id: {cid}}}) RETURN id(n) AS id',
        'MATCH (a:Entity {{community_id: {cid}}})-[r]->(b:Entity {{community_id: {cid}}}) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
    ) YIELD graphName
""")
sub_result = session.run(f"""
    CALL gds.leiden.write('ltm-subtopic-{cid}', {{
        writeProperty: 'subcommunity_id',
        relationshipWeightProperty: 'weight',
        gamma: 2.0
    }})
    YIELD communityCount, modularity
""").single()
# Only create SubTopic nodes if sub_result['communityCount'] > 1
# Create SubTopic nodes with label SubTopic, set summary via LLM
# Link: (SubTopic)-[:PART_OF]->(Topic)
# Link: (Entity {community_id: cid, subcommunity_id: X})-[:BELONGS_TO_SUB]->(SubTopic)
session.run("CALL gds.graph.drop('ltm-subtopic-{cid}', false) YIELD graphName")
```

## Entity Schema Updates
Entities already have `community_id`. Add `subcommunity_id` for the two-level case. Add `node2vec` (list of floats).

Topic nodes: add `community_hash: str` (SHA1 of sorted entity names), `entity_count: int`.

SubTopic nodes: `community_id: int` (parent), `subcommunity_id: int`, `summary: str`, `entity_count: int`.

## Return Type
`run_feature_engineering(session_id: str)` should return a dict:
```python
{
    "entity_count": int,
    "edge_count": int,
    "community_count": int,
    "modularity": float,
    "modularities": list[float],
    "ran_iterations": int,
    "did_converge": bool,
    "changed_community_ids": list[int],
}
```

## IMPORTANT PITFALLS
- `gds.graph.project.cypher` is the correct form, NOT `gds.graph.project` when using Cypher queries
- GDS graph drop: `CALL gds.graph.drop('name', false)` — second arg = failSilently, prevents error if graph does not exist
- Neo4j driver: never pass `query=` as a keyword arg to `session.run()`; use positional first arg
- `asyncio.get_running_loop()` not `get_event_loop()` when called from worker threads
- `node2vec` embedding on entities is a List<Float>; when reading back from Neo4j driver, it comes as a Python list already — no conversion needed for `gds.similarity.cosine`
- The GDSRun node stores global stats per session — it's a provenance record, not per-community
- For the two-level Leiden, use a higher `gamma` value (2.0 vs 1.0) to encourage finer sub-communities
- `gds.graph.project.cypher` may be deprecated in GDS 2.5+ in favor of `gds.graph.project` with explicit Cypher maps — check via `CALL gds.list() YIELD name WHERE name CONTAINS 'project'` and use whichever is available

## Validation
After rewriting, run:
```bash
cd /root/graphknows && PYTHONPATH=/root/graphknows/services:/root/graphknows/services/agent:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows \
/root/graphknows/services/agent/.venv/bin/python -B test_ltm_pipeline.py
```
All 6 checks must still pass. The community count should drop substantially (from 375 singletons to ~10-30 meaningful communities). Modularity should be > 0.1.
