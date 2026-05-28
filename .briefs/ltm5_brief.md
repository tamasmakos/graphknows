# LTM-5 Brief: Forget Functionality + CLI Tool (graphknows-memory) + MCP Extension

## Working Directory
/root/graphknows

## Your Tools
- neo4j-query.ts, pgvector-query.ts, memgraph-query.ts
- bash, read, grep — file inspection

## Model
claude-sonnet-4.6 via GitHub Copilot

## Context: What Exists After LTM-1/2/3/4
- Neo4j: Entity, Segment, Topic nodes; APPEARED_IN, BELONGS_TO, RELATES edges
- pgvector: ltm_topics table with community_id, summary, embedding, hub_entities
- services/graphknows/ package (check what's in it)
- services/search/mcp_server.py — existing MCP server pattern to follow

## Step 1: Inspect existing MCP server and graphknows package
```bash
head -80 /root/graphknows/services/search/mcp_server.py
ls /root/graphknows/services/graphknows/graphknows/
```

## Step 2: Create services/graphknows/graphknows/forget.py

Three-level forget implementation:

```python
"""LTM Forget — prune disconnected components, TTL decay, explicit forget."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger("graphknows.forget")


class LTMForget:
    def __init__(self, neo4j_driver: Any, pg_dsn: str) -> None:
        self._driver = neo4j_driver
        self._pg_dsn = pg_dsn

    def prune_disconnected(self, min_component_size: int = 2) -> dict:
        """Drop Entity nodes with no relations (degree=0) or tiny components."""
        with self._driver.session() as s:
            # Find isolated entities (no edges at all)
            result = s.run(
                "MATCH (e:Entity) WHERE NOT (e)--() "
                "WITH e DETACH DELETE e RETURN count(e) as deleted"
            ).single()
            deleted = result["deleted"] if result else 0
        log.info("prune_disconnected: removed %d isolated entities", deleted)
        return {"deleted_entities": deleted}

    def prune_by_ttl(self, ttl_days: int = 90, pagerank_floor: float = 0.001) -> dict:
        """Drop entities not seen in ttl_days with pagerank below floor."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
        with self._driver.session() as s:
            result = s.run(
                "MATCH (e:Entity) "
                "WHERE e.last_seen < $cutoff AND coalesce(e.pagerank, 0) < $floor "
                "DETACH DELETE e RETURN count(e) as deleted",
                cutoff=cutoff, floor=pagerank_floor,
            ).single()
            deleted = result["deleted"] if result else 0
        log.info("prune_by_ttl: removed %d stale low-rank entities", deleted)
        return {"deleted_entities": deleted}

    def forget_community(self, community_id: int) -> dict:
        """Delete all entities exclusively in this community + their Topic node."""
        import asyncio, asyncpg as _pg

        with self._driver.session() as s:
            # Only delete entities that ONLY belong to this community
            result = s.run(
                "MATCH (e:Entity {community_id: $cid}) "
                "WHERE NOT EXISTS { MATCH (e)-[:BELONGS_TO]->(t:Topic) WHERE t.community_id <> $cid } "
                "DETACH DELETE e RETURN count(e) as deleted",
                cid=community_id,
            ).single()
            deleted = result["deleted"] if result else 0
            # Delete Topic node
            s.run("MATCH (t:Topic {community_id: $cid}) DETACH DELETE t", cid=community_id)

        # Remove from pgvector
        async def _pg_delete():
            conn = await _pg.connect(self._pg_dsn)
            try:
                await conn.execute("DELETE FROM ltm_topics WHERE community_id=$1", community_id)
            finally:
                await conn.close()

        try:
            asyncio.get_event_loop().run_until_complete(_pg_delete())
        except RuntimeError:
            pass

        log.info("forget_community %d: deleted %d entities + topic node", community_id, deleted)
        return {"community_id": community_id, "deleted_entities": deleted}

    def forget_before(self, before_date: str) -> dict:
        """Delete Segment nodes older than date and orphaned entities."""
        with self._driver.session() as s:
            s.run(
                "MATCH (seg:Segment) WHERE seg.timestamp < $date DETACH DELETE seg",
                date=before_date,
            )
            # Clean up orphaned entities (no APPEARED_IN, no RELATES)
            result = s.run(
                "MATCH (e:Entity) WHERE NOT (e)-[:APPEARED_IN]->() AND NOT (e)--() "
                "DETACH DELETE e RETURN count(e) as deleted"
            ).single()
        return {"deleted_segments": True, "orphaned_entities_deleted": result["deleted"] if result else 0}
```

## Step 3: Create CLI at services/graphknows/graphknows/cli/memory.py

```python
"""graphknows-memory CLI — long-term memory management tool."""

import json
import os
import click
from neo4j import GraphDatabase


def _neo4j_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, pw))

def _pg_dsn():
    return os.environ.get("PGVECTOR_DSN", "postgresql://postgres@172.18.0.1:5432/vectordb")


@click.group()
def cli():
    """GraphKnows long-term memory management."""


@cli.command()
def stats():
    """Show LTM statistics: entity, topic, segment counts."""
    d = _neo4j_driver()
    with d.session() as s:
        ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
        tc = s.run("MATCH (t:Topic) RETURN count(t) as c").single()["c"]
        sc = s.run("MATCH (s:Segment) RETURN count(s) as c").single()["c"]
        rc = s.run("MATCH ()-[r:RELATES]->() RETURN count(r) as c").single()["c"]
        top = s.run(
            "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL "
            "RETURN e.name, e.pagerank ORDER BY e.pagerank DESC LIMIT 5"
        ).data()
    d.close()
    click.echo(f"Entities:  {ec}")
    click.echo(f"Relations: {rc}")
    click.echo(f"Topics:    {tc}")
    click.echo(f"Segments:  {sc}")
    click.echo("Top entities by PageRank:")
    for e in top:
        click.echo(f"  {e['e.name']}: {e['e.pagerank']:.4f}")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of topics to return")
def query(query, top_k):
    """Semantic search over LTM topics."""
    from sentence_transformers import SentenceTransformer
    import asyncio, asyncpg as apg

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding = json.dumps(embedder.encode(query, normalize_embeddings=True).tolist())

    async def _run():
        conn = await apg.connect(_pg_dsn())
        try:
            rows = await conn.fetch(
                "SELECT community_id, summary, entity_count FROM ltm_topics "
                f"ORDER BY embedding <=> $1::vector LIMIT {top_k}",
                embedding,
            )
            return rows
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    for r in rows:
        click.echo(f"[community {r['community_id']}] ({r['entity_count']} entities)")
        click.echo(f"  {r['summary']}")
        click.echo()


@cli.group()
def forget():
    """Forget/prune LTM entries."""


@forget.command("disconnected")
def forget_disconnected():
    """Prune isolated Entity nodes with no relations."""
    from graphknows.forget import LTMForget
    f = LTMForget(_neo4j_driver(), _pg_dsn())
    result = f.prune_disconnected()
    click.echo(f"Deleted {result['deleted_entities']} isolated entities")


@forget.command("before")
@click.argument("date")
def forget_before(date):
    """Delete Segment nodes older than DATE (ISO format: 2025-01-01)."""
    from graphknows.forget import LTMForget
    f = LTMForget(_neo4j_driver(), _pg_dsn())
    result = f.forget_before(date)
    click.echo(f"Deleted segments before {date}, orphans: {result['orphaned_entities_deleted']}")


@forget.command("community")
@click.option("--community-id", required=True, type=int)
def forget_community_cmd(community_id):
    """Delete all entities exclusively in a community."""
    from graphknows.forget import LTMForget
    f = LTMForget(_neo4j_driver(), _pg_dsn())
    result = f.forget_community(community_id)
    click.echo(f"Deleted community {result['community_id']}: {result['deleted_entities']} entities")


@cli.command()
@click.option("--skip-topics", is_flag=True)
def enrich(skip_topics):
    """Run feature engineering (Leiden + centrality) on the full LTM graph."""
    from graphknows.feature_engineering import LTMFeatureEngineer
    fe = LTMFeatureEngineer(_neo4j_driver())
    result = fe.run()
    click.echo(f"Entities: {result.entity_count}, Communities: {result.community_count}, Algorithm: {result.algorithm_used}, {result.runtime_ms:.0f}ms")
    if result.error:
        click.echo(f"Warning: {result.error}")
    if result.top_entities:
        click.echo("Top entities by PageRank:")
        for e in result.top_entities[:5]:
            click.echo(f"  {e.get('name')}: pr={e.get('pr', 0):.4f} community={e.get('community')}")
```

## Step 4: Add CLI entrypoint to pyproject.toml

Find /root/graphknows/services/graphknows/pyproject.toml and add under [project.scripts]:
```toml
graphknows-memory = "graphknows.cli.memory:cli"
```

Also create /root/graphknows/services/graphknows/graphknows/cli/__init__.py (empty).

## Step 5: Extend MCP server with 3 LTM read tools

In services/search/mcp_server.py (or new services/graphknows/graphknows/mcp_ltm.py),
add these 3 tools following the existing MCP tool pattern in the file:

Tool 1: ltm_query(query: str, top_k: int = 5) -> list[dict]
  - Embed query, ANN search ltm_topics, return [{community_id, summary, entity_count, hub_entities}]

Tool 2: ltm_entity(name: str) -> dict
  - Neo4j: MATCH (e:Entity {name:$name}) RETURN e with all properties
  - Include community, pagerank, betweenness, first_seen, last_seen, segment count

Tool 3: ltm_topics(min_entities: int = 3) -> list[dict]
  - Return all topics with entity_count >= min_entities, ordered by entity_count DESC

## Acceptance Criteria
- services/graphknows/graphknows/forget.py created and importable
- services/graphknows/graphknows/cli/memory.py created
- graphknows-memory stats runs without error (even on empty LTM)
- graphknows-memory query "EU AI regulation" returns topic results (if LTM has data)
- graphknows-memory forget --help works for all subcommands
- 3 MCP tools added and registered
- Do NOT commit — leave for principal review

## Pitfalls
- CLI must handle empty LTM gracefully (no entities/topics yet) — no crash
- forget_community must NOT delete entities that appear in other communities
- asyncpg vs psycopg2: use asyncpg for async paths, check which is installed first
- pyproject.toml location: /root/graphknows/services/graphknows/pyproject.toml (not repo root)
- After adding CLI entrypoint, install with: cd services/graphknows && pip install -e . --quiet
