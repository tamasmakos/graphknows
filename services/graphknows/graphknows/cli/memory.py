"""graphknows-memory CLI — long-term memory management tool."""

import json
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(os.environ.get("PWD", ".")) / ".env")

_NO_RESULTS = "No results found."


def _neo4j_driver():
    uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")  # NOSONAR
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    return GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR


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
    import asyncio

    import asyncpg as apg
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding = json.dumps(embedder.encode(query, normalize_embeddings=True).tolist())

    async def _run():
        conn = await apg.connect(_pg_dsn())
        try:
            rows = await conn.fetch(
                "SELECT community_id, summary, entity_count FROM ltm_topics "
                f"ORDER BY embedding <=> $1::vector LIMIT {top_k}",  # nosec B608
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
    click.echo(
        f"Entities: {result.entity_count}, Communities: {result.community_count}, Algorithm: {result.algorithm_used}, {result.runtime_ms:.0f}ms"
    )
    if result.error:
        click.echo(f"Warning: {result.error}")
    if result.top_entities:
        click.echo("Top entities by PageRank:")
        for e in result.top_entities[:5]:
            click.echo(f"  {e.get('name')}: pr={e.get('pr', 0):.4f} community={e.get('community')}")


@cli.command("recall-docs")
@click.argument("query")
@click.option("--top-k", default=5, type=int, help="Max results")
@click.option("--session", default=None, help="Filter to specific session")
def recall_docs(query: str, top_k: int, session: str | None) -> None:
    """Find document chunks matching QUERY."""
    from graphknows.recall import recall_by_document

    d = _neo4j_driver()
    results = recall_by_document(d, query, top_k, session)
    d.close()
    for r in results:
        click.echo(f"[{r.score:.2f}] {r.provenance.get('title', 'Unknown')} | {r.text[:120]}...")
        click.echo(f"  Entities: {', '.join(r.entity_names[:5])}")
    if not results:
        click.echo(_NO_RESULTS)


@cli.command("recall-dialog")
@click.argument("query")
@click.option("--top-k", default=5, type=int)
@click.option("--session", default=None)
def recall_dialog(query: str, top_k: int, session: str | None) -> None:
    """Find past conversation turns matching QUERY."""
    from graphknows.recall import recall_by_dialog

    d = _neo4j_driver()
    results = recall_by_dialog(d, query, top_k, session)
    d.close()
    for r in results:
        click.echo(
            f"[{r.score:.2f}] session={r.provenance.get('session_id', '?')} "
            f"turn={r.provenance.get('turn_index', 0)} "
            f"{r.provenance.get('role', '?')}: {r.text[:120]}..."
        )
        click.echo(f"  Entities: {', '.join(r.entity_names[:5])}")
    if not results:
        click.echo(_NO_RESULTS)


@cli.command("recall-topic")
@click.argument("query")
@click.option("--top-k", default=5, type=int)
def recall_topic(query: str, top_k: int) -> None:
    """Find topics related to QUERY."""
    from graphknows.recall import recall_by_topic

    d = _neo4j_driver()
    results = recall_by_topic(d, query, top_k)
    d.close()
    for r in results:
        click.echo(
            f"[{r.score:.2f}] {r.text[:120]} "
            f"(entities: {len(r.entity_names)}, "
            f"size: {r.provenance.get('size', 0)})"
        )
        click.echo(f"  Entities: {', '.join(r.entity_names[:5])}")
    if not results:
        click.echo(_NO_RESULTS)
