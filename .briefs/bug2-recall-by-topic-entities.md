# Brief: Bug 2 — recall_by_topic returns entities:[] and score:0.0

## Branch
`fix/mcp-bugs-round2` (already exists on origin). First step: `git checkout fix/mcp-bugs-round2`.

## Root Cause (confirmed)
`recall_by_topic()` in `graphknows/recall.py` contains this Cypher (around line 204–221):

```cypher
UNWIND $tokens AS token
MATCH (e:ENTITY)
WHERE toLower(e.name) CONTAINS token
WITH collect(DISTINCT e.name) AS matched_names
MATCH (t:TOPIC)
OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
WHERE e2.name IN matched_names
...
```

Live schema (confirmed via direct Neo4j query):
- `:IN_TOPIC` edges exist ONLY on `ENTITY_CONCEPT` nodes (10 total)
- There are 176 `ENTITY` nodes — none have `:IN_TOPIC` edges
- Tokens like "memory", "architecture", "agents" match ENTITY nodes, but NOT ENTITY_CONCEPT nodes
- Therefore `matched_names` is populated with names that never appear in the `OPTIONAL MATCH` join
- Result: `entities: []`, `score: 0`

## Fix — One File: `graphknows/recall.py`

Change line 206 from:
```cypher
MATCH (e:ENTITY)
WHERE toLower(e.name) CONTAINS token
```
to:
```cypher
MATCH (e)
WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS token
```

Everything else in the Cypher stays identical. This is a minimal, surgical change.

The full corrected Cypher block:
```python
cypher = """
UNWIND $tokens AS token
MATCH (e)
WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS token
WITH collect(DISTINCT e.name) AS matched_names
MATCH (t:TOPIC)
OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
WHERE e2.name IN matched_names
WITH t,
     collect(DISTINCT e2.name) AS matched_entities,
     count(DISTINCT e2) AS overlap
ORDER BY overlap DESC, t.entity_count DESC
LIMIT $top_k
RETURN t.community_id AS topic_id,
       t.summary AS summary,
       matched_entities AS entities,
       overlap AS score
"""
```

## Test Changes
File: `tests/retrieval/test_recall_by_topic.py` (already exists — check it first).

The test should verify:
1. The Cypher string in `recall_by_topic` contains `(e:ENTITY OR e:ENTITY_CONCEPT)` (string assertion)
2. When a mock Neo4j driver returns an ENTITY_CONCEPT name in the result, `entities` is non-empty and `score > 0`

Update or extend the existing test file. Do not add a new file.

## Acceptance Criteria
```bash
cd /root/graphknows
git checkout fix/mcp-bugs-round2
# make the change
uv run pytest tests/retrieval/test_recall_by_topic.py -v
uv run pytest tests/ -q --tb=short 2>&1 | tail-10
uv run ruff check --config ruff.toml graphknows/ tests/
uv run ruff format --check --config ruff.toml graphknows/ tests/
uv run mypy graphknows/ --strict --exclude graphgen --exclude cli/memory --exclude feature_engineering --exclude forget --exclude mcp_ltm --exclude recall --exclude topic_summarizer --ignore-missing-imports --allow-untyped-decorators --no-warn-unused-ignores 2>&1 | tail -10
git add -A
git commit -m "fix: recall_by_topic match ENTITY_CONCEPT nodes for IN_TOPIC join"
git push origin fix/mcp-bugs-round2
```

All tests must pass. No new mypy errors. Both ruff gates must pass.

## What NOT to do
- Do NOT modify any other function in `recall.py`
- Do NOT touch any other source files
- Do NOT open a GitHub PR
- Do NOT push to main
