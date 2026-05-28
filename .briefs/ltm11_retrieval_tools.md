# LTM-11: Rich Retrieval Tools — recall_by_document, recall_by_dialog, recall_by_topic as MCP+CLI

## Context

GraphKnows now has a rich LTM graph:
```
(Segment/Session)-[HAS_DIALOG]->(Dialog)-[PARTICIPANT]->(Participant)-[SAID]->(Utterance)-[MENTIONS]->(Entity)
(Document)-[HAS_CHUNK]->(Chunk)-[MENTIONS]->(Entity)
(Entity)-[BELONGS_TO]->(Topic)
(Entity)-[APPEARED_IN]->(Segment)
```

This task exposes three traversal-based recall methods:
- `recall_by_document(query, top_k)` — find Chunks mentioning entities matching a text query
- `recall_by_dialog(query, top_k)` — find Utterances from past conversations matching a query
- `recall_by_topic(query, top_k)` — find Topics semantically close to a query

Each method is exposed as:
1. A Python function in `services/graphknows/graphknows/recall.py`
2. An MCP tool in `services/graphknows/graphknows/mcp_ltm.py` (already exists — extend it)
3. A CLI subcommand in `services/graphknows/graphknows/cli/memory.py` (already exists — extend it)

## Implementation

### 1. NEW: `services/graphknows/graphknows/recall.py`

Three retrieval functions, each using Cypher traversal (no vector search required, but
use embedding cosine similarity if available):

```python
"""LTM retrieval: recall_by_document, recall_by_dialog, recall_by_topic."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RecallResult:
    source_type: str          # "document" | "dialog" | "topic"
    text: str                 # The matched text (chunk, utterance, topic label)
    entity_names: list[str]   # Entities mentioned in this text
    provenance: dict          # e.g. {document_title, url} or {session_id, turn_index}
    score: float = 1.0        # relevance score (1.0 if no embedding, cosine otherwise)


def recall_by_document(
    driver: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallResult]:
    """Find document chunks whose mentioned entities match the query.

    Strategy:
    1. Find Entity nodes whose name contains any word from query (case-insensitive)
    2. Traverse Entity ← MENTIONS ← Chunk ← HAS_CHUNK ← Document
    3. Return top_k chunks ranked by number of matched entities

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query string.
        top_k: Max results to return.
        session_filter: Optional session_id to scope to a specific session's documents.

    Returns:
        List of RecallResult ordered by relevance score.
    """
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    if not words:
        return []
    
    # Build regex pattern for entity name matching
    pattern = "|".join(words)
    
    cypher = """
        MATCH (e:Entity)
        WHERE e.name =~ $pattern
        MATCH (c:Chunk)-[:MENTIONS]->(e)
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        WITH d, c, collect(e.name) AS entities, count(e) AS match_count
        ORDER BY match_count DESC
        LIMIT $top_k
        RETURN d.title AS title, d.url AS url, c.text AS text,
               c.id AS chunk_id, entities, match_count AS score
    """
    ...


def recall_by_dialog(
    driver: Any,
    query: str,
    top_k: int = 5,
    session_filter: str | None = None,
) -> list[RecallResult]:
    """Find past utterances whose mentioned entities match the query.

    Strategy:
    1. Find Entity nodes matching query words
    2. Traverse Entity ← MENTIONS ← Utterance ← SAID ← Participant ← PARTICIPANT ← Dialog
    3. Return top_k utterances with session context

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query.
        top_k: Max results.
        session_filter: Optional session_id to filter.

    Returns:
        List of RecallResult with dialog provenance.
    """
    ...


def recall_by_topic(
    driver: Any,
    query: str,
    top_k: int = 5,
) -> list[RecallResult]:
    """Find Topics whose label or member entities match the query.

    Strategy:
    1. Find Topic nodes where label contains query words
    2. Also find Topics whose member Entities match query words
    3. Return top_k Topics with entity samples and session provenance

    Args:
        driver: Sync Neo4j driver.
        query: Free-text query.
        top_k: Max results.

    Returns:
        List of RecallResult with topic provenance.
    """
    ...
```

Implement each function fully. Use parameterized Cypher — no f-string injection of user input.
Wrap each in try/except, return empty list on error (non-fatal).

### Cypher for recall_by_document

```cypher
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($word)  -- one query per word, union results
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (d:Document)-[:HAS_CHUNK]->(c)
WITH d, c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS match_count
ORDER BY match_count DESC
LIMIT $top_k
RETURN d.title AS title, d.url AS url, c.text AS text,
       entities, toFloat(match_count) AS score
```

For multi-word queries, run one Cypher with `WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))`.

### Cypher for recall_by_dialog

```cypher
MATCH (e:Entity)
WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
MATCH (u:Utterance)-[:MENTIONS]->(e)
MATCH (p:Participant)-[:SAID]->(u)
MATCH (dia:Dialog)-[:PARTICIPANT]->(p)
WITH dia, u, p, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS match_count
ORDER BY match_count DESC, u.turn_index ASC
LIMIT $top_k
RETURN dia.session_id AS session_id, u.text AS text, u.turn_index AS turn_index,
       p.role AS role, p.name AS participant, entities, toFloat(match_count) AS score
```

### Cypher for recall_by_topic

```cypher
MATCH (t:Topic)
WHERE any(word IN $words WHERE toLower(coalesce(t.label, '')) CONTAINS toLower(word))
   OR any(word IN $words WHERE any(e IN t.top_entities WHERE toLower(e) CONTAINS toLower(word)))
WITH t, 1.0 AS score
UNION
MATCH (e:Entity)
WHERE any(word IN $words WHERE toLower(e.name) CONTAINS toLower(word))
MATCH (e)-[:BELONGS_TO]->(t:Topic)
WITH t, 0.8 AS score
RETURN t.community_id AS topic_id, coalesce(t.label, 'Topic ' + toString(t.community_id)) AS label,
       t.top_entities AS entities, t.entity_count AS size, score
ORDER BY score DESC
LIMIT $top_k
```

### 2. EXTEND: `services/graphknows/graphknows/mcp_ltm.py`

Add three new MCP tool endpoints to the existing file. Pattern (look at existing tools in the file for style):

```python
@mcp_server.tool()
def recall_by_document(query: str, top_k: int = 5) -> list[dict]:
    """Find document chunks whose entities match the query."""
    from graphknows.recall import recall_by_document as _rbd
    driver = _get_neo4j_driver()
    results = _rbd(driver, query, top_k)
    return [{"text": r.text, "entities": r.entity_names, "provenance": r.provenance, "score": r.score} for r in results]

@mcp_server.tool()
def recall_by_dialog(query: str, top_k: int = 5, session_filter: str = "") -> list[dict]:
    """Find past utterances whose entities match the query."""
    ...

@mcp_server.tool()
def recall_by_topic(query: str, top_k: int = 5) -> list[dict]:
    """Find topics semantically related to the query."""
    ...
```

Read the existing `mcp_ltm.py` first to understand the pattern (how `_get_neo4j_driver()` is defined, how tools are registered), then add at the end.

### 3. EXTEND: `services/graphknows/graphknows/cli/memory.py`

Add three new subcommands. Read the existing CLI file first for Click pattern, then add:

```python
@memory_cli.command("recall-docs")
@click.argument("query")
@click.option("--top-k", default=5, type=int, help="Max results")
@click.option("--session", default=None, help="Filter to specific session")
def recall_docs(query: str, top_k: int, session: str | None) -> None:
    """Find document chunks matching QUERY."""
    from graphknows.recall import recall_by_document
    driver = _get_driver()
    results = recall_by_document(driver, query, top_k, session)
    for r in results:
        click.echo(f"[{r.score:.2f}] {r.provenance.get('title', 'Unknown')} | {r.text[:120]}...")
        click.echo(f"  Entities: {', '.join(r.entity_names[:5])}")
    if not results:
        click.echo("No results found.")

@memory_cli.command("recall-dialog")
@click.argument("query")
@click.option("--top-k", default=5, type=int)
@click.option("--session", default=None)
def recall_dialog(query: str, top_k: int, session: str | None) -> None:
    """Find past conversation turns matching QUERY."""
    ...

@memory_cli.command("recall-topic")
@click.argument("query")
@click.option("--top-k", default=5, type=int)
def recall_topic(query: str, top_k: int) -> None:
    """Find topics related to QUERY."""
    ...
```

## Test to Add

### `test_recall.py` (NEW at repo root)

Synthetic test: create minimal Document+Chunk+Utterance fixtures, run each recall, verify results.

```python
"""Test recall_by_document, recall_by_dialog, recall_by_topic with synthetic fixtures."""
# sys.path setup...
# load .env...

from neo4j import GraphDatabase
from graphknows.recall import recall_by_document, recall_by_dialog, recall_by_topic, RecallResult

driver = GraphDatabase.driver(NEO4J_URI, auth=(user, password))

# Setup fixtures
SESSION = "test-recall-ltm11"
with driver.session() as s:
    # Entity
    s.run("MERGE (e:Entity {name: 'European Central Bank'}) SET e.type = 'INSTITUTION'")
    # Document + Chunk + MENTIONS
    s.run("""
        MERGE (d:Document {id: 'test-doc-recall'})
        SET d.title = 'ECB Policy', d.url = 'http://test.ecb.eu'
        MERGE (c:Chunk {id: 'test-chunk-recall'})
        SET c.text = 'The European Central Bank sets monetary policy for the eurozone.',
            c.char_offset = 0, c.token_count = 14
        MERGE (d)-[:HAS_CHUNK]->(c)
        MERGE (c)-[:MENTIONS]->(:Entity {name: 'European Central Bank'})
    """)
    # Segment + Dialog + Utterance + MENTIONS
    s.run("""
        MERGE (seg:Segment {session_id: $sid})
        MERGE (dia:Dialog {id: 'test-dia-recall'}) SET dia.session_id = $sid
        MERGE (seg)-[:HAS_DIALOG]->(dia)
        MERGE (p:Participant {id: 'test-p-recall'}) SET p.role = 'user', p.name = 'user'
        MERGE (dia)-[:PARTICIPANT]->(p)
        MERGE (u:Utterance {id: 'test-utt-recall'})
        SET u.text = 'What does the European Central Bank do?', u.turn_index = 0,
            u.timestamp = '2025-01-01T10:00:00Z'
        MERGE (p)-[:SAID]->(u)
        MERGE (u)-[:MENTIONS]->(:Entity {name: 'European Central Bank'})
    """, sid=SESSION)
    # Topic + BELONGS_TO
    s.run("""
        MERGE (t:Topic {community_id: 99999})
        SET t.label = 'Monetary Policy', t.entity_count = 1,
            t.top_entities = ['European Central Bank']
        MERGE (:Entity {name: 'European Central Bank'})-[:BELONGS_TO]->(t)
    """)

# Test recall_by_document
doc_results = recall_by_document(driver, "European Central Bank", top_k=5)
assert len(doc_results) >= 1, f"Expected >=1 doc result, got {len(doc_results)}"
assert any("European Central Bank" in r.entity_names for r in doc_results)
print(f"PASS  recall_by_document: {len(doc_results)} results")

# Test recall_by_dialog
dial_results = recall_by_dialog(driver, "European Central Bank", top_k=5)
assert len(dial_results) >= 1, f"Expected >=1 dialog result, got {len(dial_results)}"
print(f"PASS  recall_by_dialog: {len(dial_results)} results")

# Test recall_by_topic
topic_results = recall_by_topic(driver, "monetary policy", top_k=5)
assert len(topic_results) >= 1, f"Expected >=1 topic result, got {len(topic_results)}"
print(f"PASS  recall_by_topic: {len(topic_results)} results")

# Cleanup
with driver.session() as s:
    s.run("MATCH (d:Document {id: 'test-doc-recall'}) DETACH DELETE d")
    s.run("MATCH (c:Chunk {id: 'test-chunk-recall'}) DETACH DELETE c")
    s.run("MATCH (dia:Dialog {id: 'test-dia-recall'}) DETACH DELETE dia")
    s.run("MATCH (u:Utterance {id: 'test-utt-recall'}) DETACH DELETE u")
    s.run("MATCH (p:Participant {id: 'test-p-recall'}) DETACH DELETE p")
    s.run("MATCH (seg:Segment {session_id: $sid}) DETACH DELETE seg", sid=SESSION)
    s.run("MATCH (t:Topic {community_id: 99999}) DETACH DELETE t")

driver.close()
print("PASS  Cleanup complete")
print("All LTM-11 recall tests passed.")
```

## Acceptance Criteria

1. `recall.py` with `recall_by_document`, `recall_by_dialog`, `recall_by_topic`, `RecallResult`
2. All three functions return correct results for synthetic fixtures
3. MCP tools added to `mcp_ltm.py` (3 new tools)
4. CLI subcommands added to `cli/memory.py` (recall-docs, recall-dialog, recall-topic)
5. `test_recall.py` passes: 3/3 recall PASS + cleanup
6. Existing `test_ltm_pipeline.py` still 5/5 + schema + orphan prune green
7. Commit: `LTM-11: rich retrieval — recall_by_document, recall_by_dialog, recall_by_topic as MCP+CLI`

## Constraints

- Python 3.11, type hints, Google docstrings, 100-char lines, no bare except
- No f-string injection of user query into Cypher — always use $params
- Read existing `mcp_ltm.py` and `cli/memory.py` BEFORE modifying — follow existing patterns exactly
- All three recall functions: non-fatal, return [] on error
- Run: `cd /root/graphknows && /root/graphknows/services/agent/.venv/bin/python -B test_recall.py`
- PYTHONPATH: `/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows`
- .env at `/root/graphknows/.env`
- Do NOT commit. Leave staged for review.

## Known Facts

- `mcp_ltm.py`: `/root/graphknows/services/graphknows/graphknows/mcp_ltm.py` — read it first
- `cli/memory.py`: `/root/graphknows/services/graphknows/graphknows/cli/memory.py` — read it first
- Neo4j: bolt://172.18.0.1:7687, credentials in .env
- Topic nodes have properties: `community_id`, `label` (optional), `top_entities` (list), `entity_count`
- Entity→Topic edge is `BELONGS_TO` (from feature_engineering.py)
- Utterance→Entity edge is `MENTIONS`, Chunk→Entity edge is `MENTIONS`
- Project root: `/root/graphknows`, branch: `main`, commit: `36ba13a`
