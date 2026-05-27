"""graphknows-memory CLI — memory management: ingest, flush, recall, doctor, forget."""

from __future__ import annotations

import os

import click

from graphknows.builders import (
    build_memgraph_store,
    build_memory_ingestor,
    build_neo4j_store,
    build_pgvector_store,
)
from graphknows.config import GraphKnowsSettings

_NO_RESULTS = "No results found."


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
    s = GraphKnowsSettings()

    if is_file:
        ingestor = build_memory_ingestor(s)
        result = ingestor.ingest_file(text_or_path, session_id=session)
    else:
        from graphknows.memory.extractor import GLiNER2EntityExtractor
        from graphknows.memory.ingestor import MemoryIngestor, _run
        from graphknows.retrieval.stores.embedder import embed_one

        mg = build_memgraph_store(s)
        pg = build_pgvector_store(s)
        _run(pg.connect())
        extractor = GLiNER2EntityExtractor()
        ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=embed_one)
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

    s = GraphKnowsSettings()
    mg = build_memgraph_store(s)
    pg = build_pgvector_store(s)
    neo = build_neo4j_store(s)

    key = s.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")  # nosec
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
    from graphknows.retrieval.stores.embedder import embed_one

    s = GraphKnowsSettings()
    mg = build_memgraph_store(s)
    pg = build_pgvector_store(s)
    _run(pg.connect())
    results = recall_short_term(mg, pg, query, session_id=session, top_k=top_k, embed_fn=embed_one)

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
    from graphknows.retrieval.stores.embedder import embed_one

    s = GraphKnowsSettings()
    pg = build_pgvector_store(s)
    _run(pg.connect())
    neo = build_neo4j_store(s)
    results = recall_long_term(
        neo, pg, query, top_k=top_k, session_filter=session, embed_fn=embed_one
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

    from graphknows.memory.adapter import GraphknowsMemoryAdapter

    s = GraphKnowsSettings()

    async def _get_stats():
        adapter = GraphknowsMemoryAdapter(s)
        return await adapter.stats(session or "")

    stats = asyncio.run(_get_stats())
    click.echo("pgvector")
    click.echo(f"  stm_chunks: {stats.stm.get('chunks', 0)}")
    click.echo(f"  stm_turns:  {stats.stm.get('turns', 0)}")
    click.echo(f"  ltm_chunks: {stats.ltm.get('chunks', 0)}")
    click.echo(f"  ltm_turns:  {stats.ltm.get('turns', 0)}")
    click.echo(f"  ltm_topics: {stats.ltm.get('topics', 0)}")
    click.echo("neo4j")
    click.echo(f"  sessions: {stats.ltm.get('sessions', 0)}")
    click.echo(f"  entities: {stats.ltm.get('entities', 0)}")
    click.echo(f"  topics:   {stats.ltm.get('topics', 0)}")


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

    s = GraphKnowsSettings()
    mg = build_memgraph_store(s)
    pg = build_pgvector_store(s)

    mg.delete_session(session_id)
    asyncio.run(pg.clear_stm(session_id))
    click.echo(f"Deleted STM for session {session_id}")

    if both:
        neo = build_neo4j_store(s)
        with neo._driver.session() as ns:  # nosec
            ns.run(
                "MATCH (sess:SESSION {id: $sid}) "
                "OPTIONAL MATCH (sess)-[*..4]->(n) "
                "DETACH DELETE sess, n",
                sid=session_id,
            )
        neo.close()
        click.echo(f"Deleted LTM for session {session_id}")


@forget.command("before")
@click.argument("date")
def forget_before(date: str) -> None:
    """Delete SESSION nodes flushed before DATE (ISO format: 2025-01-01)."""
    s = GraphKnowsSettings()
    neo = build_neo4j_store(s)
    with neo._driver.session() as ns:  # nosec
        r = ns.run(
            "MATCH (sess:SESSION) WHERE sess.flushed_at < $dt "
            "OPTIONAL MATCH (sess)-[*..4]->(n) "
            "DETACH DELETE sess, n "
            "RETURN count(sess) AS deleted",
            dt=date,
        ).single()
    neo.close()
    click.echo(f"Deleted {r['deleted']} sessions before {date}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@cli.command("doctor")
def doctor_cmd() -> None:
    """Health check: connectivity to Memgraph, Neo4j, pgvector."""
    import asyncio

    from graphknows.config.health import check_backends

    result = asyncio.run(check_backends(GraphKnowsSettings()))
    for key, val in result.items():
        status = "[OK]" if val is True else "[FAIL]" if val is False else "[INFO]"
        click.echo(f"{status} {key}: {val}")
