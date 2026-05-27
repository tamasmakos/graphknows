"""graphknows-memory CLI — memory management: ingest, flush, recall, doctor, forget."""

from __future__ import annotations

import os
from pathlib import Path

import click
from graphknows.config import GraphKnowsSettings

_NO_RESULTS = "No results found."

_settings: GraphKnowsSettings | None = None


def _get_settings() -> GraphKnowsSettings:
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def _neo4j_driver():
    from neo4j import GraphDatabase

    s = _get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))  # NOSONAR


def _pg_dsn() -> str:
    return _get_settings().pgvector_dsn


def _mg_uri() -> str:
    return _get_settings().memgraph_uri


def _embed_fn():
    from graphknows.retrieval.stores.embedder import embed_one

    return embed_one


@click.group()
def cli() -> None:
    """GraphKnows memory management — ingest, flush, recall, forget."""


# ---------------------------------------------------------------------------
# memory ingest
# ---------------------------------------------------------------------------


@cli.command("ingest")
@click.argument("text_or_path")
@click.option("--session", required=True, help="Session ID for STM namespace")
@click.option("--title", default="", help="Optional document title")
@click.option("--is-file", is_flag=True, help="Treat argument as a file path")
def ingest_cmd(text_or_path: str, session: str, title: str, is_file: bool) -> None:
    """Ingest TEXT_OR_PATH into short-term memory (Memgraph + pgvector stm_*).

    Accepts raw text directly or a file path (with --is-file).
    Parser is selected automatically based on file extension.
    """
    from graphknows.memory.ingestor import MemoryIngestor, _run
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())
    _run(pg.connect())  # connect inside _BG_LOOP so pool is on the right event loop
    from graphknows.memory.extractor import GLiNER2EntityExtractor

    extractor = GLiNER2EntityExtractor()
    ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=_embed_fn())

    if is_file:
        result = ingestor.ingest_file(text_or_path, session_id=session)
    else:
        result = ingestor.ingest_text(text_or_path, session_id=session, title=title)

    click.echo(f"Ingested: file_id={result.file_id}")
    click.echo(f"  chunks:   {result.chunk_count}")
    click.echo(f"  entities: {result.entity_count}")
    click.echo(f"  elapsed:  {result.elapsed_s:.2f}s")
    if result.errors:
        click.echo(f"  errors:   {len(result.errors)}")
        for e in result.errors[:3]:
            click.echo(f"    - {e}")


# ---------------------------------------------------------------------------
# memory flush
# ---------------------------------------------------------------------------


@cli.command("flush")
@click.argument("session_id")
@click.option("--no-topics", is_flag=True, help="Skip LLM topic summarization")
@click.option("--no-clear", is_flag=True, help="Keep STM after flush")
@click.option("--gds", is_flag=True, help="Run GDS PageRank + betweenness")
def flush_cmd(session_id: str, no_topics: bool, no_clear: bool, gds: bool) -> None:
    """Flush SESSION_ID from short-term memory (Memgraph/stm_*) into Neo4j/ltm_*.

    Optionally runs topic summarization (Louvain + LLM) and GDS enrichment.
    """
    import asyncio

    from graphknows.memory.flush import FlushOptions, SessionFlusher
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())

    from graphknows.retrieval.stores.neo4j import Neo4jStore

    neo = Neo4jStore(
        uri=os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "testpassword123"),
    )

    key = os.environ.get("OPENROUTER_API_KEY", "")
    flusher = SessionFlusher(mg, pg, neo, openrouter_key=key)
    opts = FlushOptions(
        summarize_topics=not no_topics,
        run_gds=gds,
        clear_stm=not no_clear,
        merge_entities=True,
    )

    async def _run_flush():
        await pg.connect()
        return await flusher.flush(session_id, opts)

    result = asyncio.run(_run_flush())
    click.echo(f"Flush session={session_id}")
    click.echo(f"  nodes:    {result.nodes_written}")
    click.echo(f"  edges:    {result.edges_written}")
    click.echo(f"  chunks:   {result.chunks_promoted}")
    click.echo(f"  turns:    {result.turns_promoted}")
    click.echo(f"  topics:   {result.topics_created}")
    click.echo(f"  merged:   {result.merged_entities}")
    click.echo(f"  elapsed:  {result.elapsed_s:.2f}s")
    if result.errors:
        click.echo(f"  errors:   {len(result.errors)}")
        for e in result.errors[:3]:
            click.echo(f"    - {e}")


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


@cli.command("recall-short")
@click.argument("query")
@click.option("--session", required=True, help="Session ID to scope recall")
@click.option("--top-k", default=5, type=int, help="Max results")
def recall_short_cmd(query: str, session: str, top_k: int) -> None:
    """Recall from short-term memory (Memgraph + stm_chunks/stm_turns).

    Results are RRF-fused from ANN vector search and entity-first graph traversal.
    """
    from graphknows.memory.ingestor import _run
    from graphknows.retrieval.recall_stm import recall_short_term
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())
    _run(pg.connect())
    results = recall_short_term(
        mg, pg, query, session_id=session, top_k=top_k, embed_fn=_embed_fn()
    )

    if not results:
        click.echo(_NO_RESULTS)
        return
    for r in results:
        prefix = f"[{r.source_type.upper()} {r.score:.4f}]"
        loc = r.heading_path or r.provenance.get("role", "")
        click.echo(f"{prefix} {loc}")
        click.echo(f"  {r.text[:200]}")
        if r.entity_names:
            click.echo(f"  entities: {', '.join(r.entity_names[:5])}")
        click.echo()


@cli.command("recall-long")
@click.argument("query")
@click.option("--top-k", default=5, type=int, help="Max results")
@click.option("--session", default=None, help="Optional session filter")
def recall_long_cmd(query: str, top_k: int, session: str | None) -> None:
    """Recall from long-term memory (Neo4j + ltm_chunks/ltm_turns).

    Entity-first graph traversal + ANN vector search, RRF-fused.
    """
    from graphknows.memory.ingestor import _run
    from graphknows.retrieval.recall_ltm import recall_long_term
    from graphknows.retrieval.stores.neo4j import Neo4jStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    pg = PgVectorStore(_pg_dsn())
    _run(pg.connect())
    neo = Neo4jStore(
        uri=os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "testpassword123"),
    )
    results = recall_long_term(
        neo, pg, query, top_k=top_k, session_filter=session, embed_fn=_embed_fn()
    )
    neo.close()

    if not results:
        click.echo(_NO_RESULTS)
        return
    for r in results:
        prefix = f"[{r.source_type.upper()} {r.score:.4f}]"
        loc = r.heading_path or r.provenance.get("session_id", "")
        click.echo(f"{prefix} {loc}")
        click.echo(f"  {r.text[:200]}")
        if r.entity_names:
            click.echo(f"  entities: {', '.join(r.entity_names[:5])}")
        click.echo()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.option("--session", default=None, help="Session to inspect")
def stats_cmd(session: str | None) -> None:
    """Show memory statistics: STM and LTM node/edge counts."""
    import asyncio

    import asyncpg as apg

    async def _pg_counts() -> dict:
        conn = await apg.connect(_pg_dsn())
        try:
            sc = await conn.fetchval("SELECT count(*) FROM stm_chunks")
            st = await conn.fetchval("SELECT count(*) FROM stm_turns")
            lc = await conn.fetchval("SELECT count(*) FROM ltm_chunks")
            lt = await conn.fetchval("SELECT count(*) FROM ltm_turns")
            top = await conn.fetchval("SELECT count(*) FROM ltm_topics")
            return {
                "stm_chunks": sc,
                "stm_turns": st,
                "ltm_chunks": lc,
                "ltm_turns": lt,
                "ltm_topics": top,
            }
        finally:
            await conn.close()

    pg_counts = asyncio.run(_pg_counts())
    click.echo("pgvector")
    for k, v in pg_counts.items():
        click.echo(f"  {k}: {v}")

    d = _neo4j_driver()
    with d.session() as s:
        ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
        tc = s.run("MATCH (t:TOPIC) RETURN count(t) as c").single()["c"]
        sess_c = s.run("MATCH (s:SESSION) RETURN count(s) as c").single()["c"]
    d.close()
    click.echo("neo4j")
    click.echo(f"  sessions: {sess_c}")
    click.echo(f"  entities: {ec}")
    click.echo(f"  topics:   {tc}")


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


@cli.group()
def forget() -> None:
    """Prune / delete memory entries."""


@forget.command("session")
@click.argument("session_id")
@click.option("--both", is_flag=True, help="Delete from LTM (Neo4j) too, not just STM")
def forget_session(session_id: str, both: bool) -> None:
    """Delete all STM data for SESSION_ID. With --both also deletes from LTM."""
    import asyncio

    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())

    mg.delete_session(session_id)
    asyncio.run(pg.clear_stm(session_id))
    click.echo(f"Deleted STM for session {session_id}")

    if both:
        d = _neo4j_driver()
        with d.session() as s:
            s.run(
                "MATCH (sess:SESSION {id: $sid}) "
                "OPTIONAL MATCH (sess)-[*..4]->(n) "
                "DETACH DELETE sess, n",
                sid=session_id,
            )
        d.close()
        click.echo(f"Deleted LTM for session {session_id}")


@forget.command("before")
@click.argument("date")
def forget_before(date: str) -> None:
    """Delete SESSION nodes flushed before DATE (ISO format: 2025-01-01)."""
    d = _neo4j_driver()
    with d.session() as s:
        r = s.run(
            "MATCH (sess:SESSION) WHERE sess.flushed_at < $dt "
            "OPTIONAL MATCH (sess)-[*..4]->(n) "
            "DETACH DELETE sess, n "
            "RETURN count(sess) AS deleted",
            dt=date,
        ).single()
    d.close()
    click.echo(f"Deleted {r['deleted']} sessions before {date}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@cli.command("doctor")
def doctor_cmd() -> None:
    """Health check: connectivity to Memgraph, Neo4j, pgvector."""
    import asyncio

    import asyncpg as apg

    # Memgraph
    try:
        from neo4j import GraphDatabase

        mg_drv = GraphDatabase.driver(_mg_uri(), auth=("", ""))  # NOSONAR
        with mg_drv.session() as s:
            s.run("RETURN 1").single()
        mg_drv.close()
        click.echo("[OK] Memgraph reachable")
    except Exception as exc:
        click.echo(f"[FAIL] Memgraph: {exc}")

    # Neo4j
    try:
        d = _neo4j_driver()
        with d.session() as s:
            s.run("RETURN 1").single()
        d.close()
        click.echo("[OK] Neo4j reachable")
    except Exception as exc:
        click.echo(f"[FAIL] Neo4j: {exc}")

    # pgvector
    async def _pg_check():
        conn = await apg.connect(_pg_dsn())
        await conn.close()

    try:
        asyncio.run(_pg_check())
        click.echo("[OK] pgvector reachable")
    except Exception as exc:
        click.echo(f"[FAIL] pgvector: {exc}")
