# LTM-4 Brief: Agent recall_node — query LTM at session start, inject prior knowledge

## Working Directory
/root/graphknows

## Your Tools
- neo4j-query.ts  → Cypher against Neo4j bolt://172.18.0.1:7687
- pgvector-query.ts → SQL against PostgreSQL 172.18.0.1:5432
- bash, read, grep — standard file inspection

## Model
claude-sonnet-4.6 via GitHub Copilot

## Context: What Exists After LTM-1/2/3
- Neo4j has Entity nodes with pagerank + community_id + BELONGS_TO Topic edges
- pgvector ltm_topics table: community_id, summary, embedding (dim=384)
- services/agent/graph/state.py: InvestigativeState TypedDict at line 85, has node_features field at line 101
- services/agent/graph/agent.py: build_graph() function that constructs the LangGraph StateGraph

## Step 1: Inspect current graph wiring
Read services/agent/graph/agent.py and find:
- The build_graph() function
- How nodes are added to the graph (add_node, add_edge)
- The current node sequence (scope → investigate → communicate or similar)
- How nodes are defined (async functions or classes)

Read services/agent/graph/state.py fully to understand InvestigativeState fields.

## Step 2: Add prior_knowledge field to InvestigativeState

In services/agent/graph/state.py, find the InvestigativeState TypedDict definition and add:
```python
prior_knowledge: str  # LTM recall result injected at session start
```
Also update the default factory / initial state function to include:
```python
prior_knowledge=""
```

## Step 3: Create services/agent/graph/nodes/recall.py

```python
"""recall_node — query LTM at session start and inject prior knowledge into state."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agent.graph.state import InvestigativeState

log = logging.getLogger("agent.graph.nodes.recall")

MAX_TOPICS = 5
MAX_HUB_ENTITIES = 3
MAX_PRIOR_CHARS = 1800  # hard cap — prior_knowledge is injected into LLM prompt


def _get_embedder():
    """Lazy singleton for BAAI/bge-small-en-v1.5."""
    if not hasattr(_get_embedder, "_model"):
        from sentence_transformers import SentenceTransformer
        _get_embedder._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _get_embedder._model


async def recall_node(state: InvestigativeState) -> InvestigativeState:
    """Query LTM topics semantically and inject top-K as prior_knowledge.

    Degrades gracefully if LTM is empty or unavailable.
    Must complete in < 2 seconds.
    """
    query = state.get("topic") or ""
    if not query:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            query = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")

    if not query:
        log.debug("recall_node: no query — skipping")
        return {**state, "prior_knowledge": ""}

    try:
        import asyncpg

        pg_dsn = os.environ.get("PGVECTOR_DSN", "")
        if not pg_dsn:
            log.debug("recall_node: PGVECTOR_DSN not set — skipping")
            return {**state, "prior_knowledge": ""}

        # 1. Embed query
        embedding = _get_embedder().encode(query, normalize_embeddings=True).tolist()
        embedding_str = json.dumps(embedding)

        # 2. ANN search over ltm_topics
        pg = await asyncpg.connect(pg_dsn, timeout=5)
        try:
            rows = await pg.fetch(
                "SELECT community_id, summary, entity_count, hub_entities "
                "FROM ltm_topics "
                f"ORDER BY embedding <=> $1::vector LIMIT {MAX_TOPICS}",
                embedding_str,
            )
        finally:
            await pg.close()

        if not rows:
            log.debug("recall_node: ltm_topics empty — skipping")
            return {**state, "prior_knowledge": ""}

        # 3. Fetch hub entity names from Neo4j for each community
        from neo4j import AsyncGraphDatabase
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")

        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
        topic_blocks = []
        try:
            async with driver.session() as s:
                for row in rows:
                    cid = row["community_id"]
                    summary = row["summary"]
                    # hub_entities stored as JSON array in pgvector
                    hub = json.loads(row["hub_entities"] or "[]")[:MAX_HUB_ENTITIES]
                    topic_blocks.append(
                        f"- {summary} [key entities: {', '.join(hub)}]"
                    )
        finally:
            await driver.close()

        # 4. Build prior_knowledge string
        prior = "Relevant prior research from long-term memory:\n" + "\n".join(topic_blocks)
        if len(prior) > MAX_PRIOR_CHARS:
            prior = prior[:MAX_PRIOR_CHARS] + "..."

        log.info("recall_node: injecting %d topic clusters (%d chars)", len(rows), len(prior))
        return {**state, "prior_knowledge": prior}

    except Exception:
        log.exception("recall_node failed — degrading gracefully")
        return {**state, "prior_knowledge": ""}
```

## Step 4: Wire recall_node into build_graph()

In services/agent/graph/agent.py, find build_graph() and:

1. Import recall_node at top:
```python
from agent.graph.nodes.recall import recall_node
```

2. In the graph assembly, add recall_node between scope_node and the first investigate/search node:
```python
graph.add_node("recall", recall_node)
# Change the edge from scope -> investigate to scope -> recall -> investigate:
# (find the existing scope -> first_node edge and insert recall between them)
graph.add_edge("scope", "recall")
graph.add_edge("recall", "investigate")  # or whatever the next node is called
```

3. If there is no "scope" node, add recall_node as the entry point before the main agent loop.

## Step 5: Inject prior_knowledge into system prompt

In services/agent/graph/nodes/ find the communicate or scope node (wherever the system prompt is built).
Add a block that prepends prior_knowledge when non-empty:

```python
prior = state.get("prior_knowledge", "")
if prior:
    system_prompt = prior + "\n\n---\nFocus new research on gaps not covered above.\n\n" + system_prompt
```

## Step 6: Verify

After wiring, import check:
```bash
PYTHONPATH=/root/graphknows/services python3 -B -c   "from agent.graph.nodes.recall import recall_node; print('recall_node OK')"
```

Also verify InvestigativeState has prior_knowledge:
```bash
PYTHONPATH=/root/graphknows/services python3 -B -c   "from agent.graph.state import InvestigativeState; print('prior_knowledge' in InvestigativeState.__annotations__)"
```

## Acceptance Criteria
- services/agent/graph/nodes/recall.py created and importable
- prior_knowledge field in InvestigativeState
- recall_node wired in build_graph() between scope and investigate
- recall_node degrades silently when ltm_topics is empty (no exception, no delay > 2s)
- prior_knowledge truncated to MAX_PRIOR_CHARS=1800 chars maximum
- Do NOT commit — leave for principal review

## Pitfalls
- asyncpg.connect timeout=5 — never block indefinitely on DB connect in an agent node
- pgvector vector cast: embedding must be a JSON string cast to ::vector, not a Python list
- InvestigativeState is a TypedDict — add prior_knowledge: str with default "" in the factory function too
- recall_node must return a dict (partial state update), not modify state in-place
- If no scope_node exists, find the actual entry node name in build_graph() before wiring
