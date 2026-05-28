# Brief: LTM-7 — Quality metrics, GDSRun fix, graph cleanup, fresh test run

## Working directory
`/root/graphknows`

## Python binary
`/root/graphknows/services/agent/.venv/bin/python`

## PYTHONPATH
`/root/graphknows/services:/root/graphknows/services/agent:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows`

## dotenv
Load from `/root/graphknows/.env`. NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are the relevant vars.

---

## Objective
Four coordinated changes, all in one pass:

### Fix 1: GDSRun always written (feature_engineering.py)

File: `services/graphknows/graphknows/feature_engineering.py`

Current bug: line ~248 has `if session_id:` — so GDSRun is never created when session_id is empty string.

Fix: change to use a fallback ID:
```python
run_id = session_id or f"global-{int(time.time())}"
# then use run_id for the MERGE
s.run(
    "MERGE (r:GDSRun {session_id: $session_id}) SET ...",
    session_id=run_id, ...
)
```

### Fix 2: Quality metrics computation (feature_engineering.py)

After Leiden writes `community_id` to all entities, add a new function `_compute_quality_metrics(s, result)` called within `run_feature_engineering` that:

1. **Silhouette score on Topic communities** using entity node2vec embeddings:
   ```python
   import numpy as np
   from sklearn.metrics import silhouette_score

   # Fetch entity embeddings and community labels
   rows = s.run(
       "MATCH (e:Entity) WHERE e.node2vec IS NOT NULL AND e.community_id IS NOT NULL "
       "RETURN e.node2vec as emb, e.community_id as cid"
   ).data()

   if len(rows) >= 10:
       X = np.array([r['emb'] for r in rows], dtype=np.float32)
       labels = np.array([r['cid'] for r in rows])
       # Filter: only communities with >= 2 members (silhouette requires it)
       from collections import Counter
       counts = Counter(labels)
       mask = np.array([counts[l] >= 2 for l in labels])
       X_filt = X[mask]
       labels_filt = labels[mask]
       if len(np.unique(labels_filt)) >= 2 and len(X_filt) >= 4:
           sil = float(silhouette_score(X_filt, labels_filt, metric='cosine'))
       else:
           sil = None
   ```

2. **Global separation and global overlap** (from thesis, section 4.4):
   - Compute per-community centroid = mean of entity embeddings in that community
   - For each community pair (A, B): if any entity in A is closer to centroid_B than centroid_A → overlap
   - global_separation = fraction of entities closer to own centroid than any other centroid
   - global_overlap = 1 - global_separation

   Simplified implementation:
   ```python
   # community centroids
   from collections import defaultdict
   comm_vecs = defaultdict(list)
   for r in rows:
       if counts[r['cid']] >= 2:
           comm_vecs[r['cid']].append(np.array(r['emb'], dtype=np.float32))
   
   centroids = {cid: np.mean(vecs, axis=0) for cid, vecs in comm_vecs.items()}
   cid_list = list(centroids.keys())
   centroid_matrix = np.stack([centroids[c] for c in cid_list])  # shape (C, D)
   
   separated = 0
   total = 0
   for r in rows:
       cid = r['cid']
       if cid not in centroids:
           continue
       emb = np.array(r['emb'], dtype=np.float32)
       # cosine distance to each centroid
       norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(emb)
       norms = np.where(norms == 0, 1e-9, norms)
       sims = centroid_matrix @ emb / norms
       own_idx = cid_list.index(cid)
       own_sim = sims[own_idx]
       if all(own_sim >= s_ for s_ in sims):  # closest to own centroid
           separated += 1
       total += 1
   
   global_separation = separated / total if total > 0 else None
   global_overlap = 1.0 - global_separation if global_separation is not None else None
   ```

3. **Unweighted modularity** — run `gds.leiden.stats` (not write) without weight property to get baseline:
   ```python
   unweighted_result = s.run("""
       CALL gds.leiden.stats('ltm-entities', {gamma: 1.0, theta: 0.01})
       YIELD modularity
   """).single()
   unweighted_modularity = float(unweighted_result['modularity']) if unweighted_result else None
   ```
   NOTE: You must call this BEFORE dropping the projected graph. Also note: the graph at this point already has weights in the projection. To get truly unweighted modularity, use gds.louvain.stats or re-project without weights. Simplification: just run `gds.louvain.stats` on the same graph without `relationshipWeightProperty`:
   ```python
   try:
       unweighted_row = s.run("""
           CALL gds.louvain.stats('ltm-entities', {gamma: 1.0})
           YIELD modularity
       """).single()
       unweighted_modularity = float(unweighted_row['modularity']) if unweighted_row else None
   except Exception:
       unweighted_modularity = None
   ```

4. Store ALL quality metrics on GDSRun node:
   ```python
   s.run(
       "MERGE (r:GDSRun {session_id: $sid}) SET "
       "r.ran_at = datetime(), "
       "r.community_count = $cc, "
       "r.modularity = $mod, "  # weighted
       "r.unweighted_modularity = $umod, "
       "r.modularities = $mods, "
       "r.ran_iterations = $iters, "
       "r.did_converge = $conv, "
       "r.entity_count = $ec, "
       "r.edge_count = $edgec, "
       "r.silhouette_cosine = $sil, "
       "r.global_separation = $gsep, "
       "r.global_overlap = $goverlap",
       sid=run_id, cc=community_count, mod=modularity, umod=unweighted_modularity,
       mods=modularities, iters=ran_iterations, conv=did_converge,
       ec=ec, edgec=edge_count, sil=sil, gsep=global_separation, goverlap=global_overlap
   )
   ```

5. Also add these to the return dict:
   ```python
   result["silhouette_cosine"] = sil
   result["global_separation"] = global_separation
   result["global_overlap"] = global_overlap
   result["unweighted_modularity"] = unweighted_modularity
   ```

### Fix 3: Clean stale Neo4j data + wipe old Topic nodes (standalone script)

Create a script `/root/graphknows/clean_ltm_graph.py` that:

1. Loads .env
2. Deletes all `Topic` nodes and their relationships (old corpus, wrong community IDs)
3. Deletes all `SubTopic` nodes and relationships
4. Deletes all `GDSRun` nodes
5. Sets `community_id = null` on all Entity nodes (so Leiden can write fresh IDs)
6. Keeps Entity nodes, Segment nodes, APPEARED_IN edges, entity-entity edges, node2vec embeddings, weights (these are fine)
7. Prints counts before and after

```python
with d.session() as s:
    s.run("MATCH (t:Topic) DETACH DELETE t")
    s.run("MATCH (t:SubTopic) DETACH DELETE t")
    s.run("MATCH (r:GDSRun) DELETE r")
    s.run("MATCH (e:Entity) REMOVE e.community_id")
    s.run("MATCH (e:Entity) REMOVE e.subcommunity_id")
    # Verify
    ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()['c']
    tc = s.run("MATCH (t:Topic) RETURN count(t) as c").single()['c']
    print(f"Entities: {ec}, Topics: {tc} (should be 0)")
```

### Fix 4: Update test_ltm_pipeline.py to pass session_id and print quality metrics

File: `/root/graphknows/test_ltm_pipeline.py`

Find the call to `run_feature_engineering` or `merge_session_to_neo4j` and ensure it passes a real session_id string.

Also find the FINAL STATE section and add printing of quality metrics from GDSRun:
```python
# After all sessions run, fetch and print GDSRun quality metrics
with neo4j_driver.session() as s:
    runs = s.run(
        "MATCH (r:GDSRun) RETURN r ORDER BY r.ran_at"
    ).data()
    print("\nGDS Quality Metrics per run:")
    for row in runs:
        r = dict(row['r'])
        print(f"  session={r.get('session_id')} communities={r.get('community_count')} "
              f"modularity(weighted)={round(r.get('modularity') or 0, 4)} "
              f"modularity(unweighted)={round(r.get('unweighted_modularity') or 0, 4)} "
              f"silhouette={round(r.get('silhouette_cosine') or 0, 4)} "
              f"global_sep={round(r.get('global_separation') or 0, 4)} "
              f"global_overlap={round(r.get('global_overlap') or 0, 4)}")
```

---

## Execution order

1. Read current `feature_engineering.py` and `test_ltm_pipeline.py` in full
2. Apply Fix 1 and Fix 2 to `feature_engineering.py`
3. Create `clean_ltm_graph.py`
4. Apply Fix 4 to `test_ltm_pipeline.py`
5. Run cleanup: `python -B clean_ltm_graph.py`
6. Run test: `python -B test_ltm_pipeline.py`
7. Report: print the full GDS Quality Metrics section from the test output

## Pitfalls
- sklearn is in the agent venv: `/root/graphknows/services/agent/.venv/bin/pip show scikit-learn` — verify first
- numpy is in the agent venv — also verify
- `gds.louvain.stats` may not accept `gamma` — try without it if it errors
- The `_compute_quality_metrics` call must happen BEFORE the `finally: gds.graph.drop(...)` block
- Do NOT delete Entity nodes, Segment nodes, APPEARED_IN edges, or entity-entity edges in the cleanup script
- The GDSRun MERGE must use `run_id` (the fallback variable), not `session_id` directly
- `silhouette_score` from sklearn requires at least 2 unique labels in filtered data; guard with try/except
- `np.linalg.norm` on a zero vector → division by zero → use `np.where(norms == 0, 1e-9, norms)`
- After cleanup, re-run feature engineering by running test_ltm_pipeline.py (it calls merge_session_to_neo4j which calls feature engineering)
