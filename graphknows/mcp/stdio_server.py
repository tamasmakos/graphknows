r"""Standalone MCP stdio server for graphknows memory operations.

Exposes 11 tools over JSON-RPC 2.0 via stdin/stdout so any MCP-aware client
(Claude Desktop, Cursor, Zed, Hermes, OpenCode) can use graphknows memory
without requiring the full agent service.

Tools
-----
STM tools:
  memory_doctor   — connectivity check: memgraph, neo4j, pgvector
  memory_stats    — STM and LTM node/chunk counts
  memory_ingest   — ingest text into STM (Memgraph + pgvector)
  memory_recall   — hybrid RRF recall from STM, LTM, or both
  memory_flush    — promote STM session → Neo4j LTM

LTM tools:
  ltm_query             — semantic topic search (pgvector)
  ltm_entity            — named entity lookup in Neo4j LTM
  ltm_topics            — list topics by entity threshold
  recall_by_document    — document chunks by entity match
  recall_by_dialog      — conversation utterances by entity match
  recall_by_topic       — topic matching by entity relevance

Usage (stdio probe)::

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
      | graphknows-mcp 2>/dev/null

All MCP protocol traffic goes to stdout. All logging goes to stderr.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env — prefer GRAPHKNOWS_ENV_FILE if set (explicit), else PWD/.env
_env_file = os.environ.get("GRAPHKNOWS_ENV_FILE") or str(Path(os.environ.get("PWD", ".")) / ".env")
load_dotenv(_env_file, override=False)

logger = logging.getLogger("graphknows.mcp.stdio")
logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)

# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

app = FastMCP(
    "graphknows-memory",
    instructions=(
        "graphknows-memory exposes short-term and long-term knowledge-graph "
        "backed memory. Use memory_ingest to store content, memory_recall to "
        "retrieve it, memory_flush to promote session memory to long-term, "
        "memory_doctor to verify backend connectivity, and memory_stats for counts. "
        "For LTM exploration: ltm_query searches topics semantically, "
        "ltm_entity looks up named entities, ltm_topics lists topics by threshold. "
        "recall_by_document, recall_by_dialog, and recall_by_topic provide "
        "entity-matched retrieval across documents, conversations, and topics."
    ),
)

# ---------------------------------------------------------------------------
# Backend helpers — re-use the same env-var pattern as cli/memory.py
# ---------------------------------------------------------------------------


def _mg_uri() -> str:
    return os.environ.get("MEMGRAPH_URI", "bolt://172.18.0.1:17687")


def _pg_dsn() -> str:
    return os.environ.get(
        "PGVECTOR_DSN",
        "postgresql://postgres:***@172.18.0.1:5432/graphknows",
    )


def _neo4j_creds() -> tuple[str, str, str]:
    uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")  # NOSONAR
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw = os.environ.get("NEO4J_PASSWORD", "testpassword123")
    return uri, user, pw


def _embed_fn() -> Any:
    from graphknows.retrieval.stores.embedder import embed_one

    return embed_one


# ---------------------------------------------------------------------------
# Lazy singleton stores — initialised on first tool call, not at import time
# ---------------------------------------------------------------------------

_mg: Any = None
_pg: Any = None
_pg_connected: bool = False


def _get_mg():
    global _mg
    if _mg is None:
        from graphknows.retrieval.stores.memgraph import MemgraphStore

        _mg = MemgraphStore(_mg_uri())
    return _mg


async def _get_pg():
    global _pg, _pg_connected
    if _pg is None:
        from graphknows.retrieval.stores.pgvector import PgVectorStore

        _pg = PgVectorStore(_pg_dsn())
    if not _pg_connected:
        await _pg.connect()
        _pg_connected = True
    return _pg


def _get_neo():
    from graphknows.retrieval.stores.neo4j import Neo4jStore

    uri, user, pw = _neo4j_creds()
    return Neo4jStore(uri=uri, user=user, password=pw)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@app.tool()
async def memory_doctor() -> dict[str, Any]:
    """Check graphknows backend connectivity (Memgraph, Neo4j, pgvector).

    Returns a dict with boolean values for each backend, plus version and pid.
    """
    import subprocess

    import asyncpg

    result: dict[str, Any] = {}

    # Memgraph
    try:
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(_mg_uri(), auth=("", ""))  # NOSONAR
        with drv.session() as s:
            s.run("RETURN 1").single()
        drv.close()
        result["memgraph"] = True
    except Exception:
        logger.exception("Memgraph connectivity check failed")
        result["memgraph"] = False

    # Neo4j
    try:
        from neo4j import GraphDatabase

        uri, user, pw = _neo4j_creds()
        drv = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
        with drv.session() as s:
            s.run("RETURN 1").single()
        drv.close()
        result["neo4j"] = True
    except Exception:
        logger.exception("Neo4j connectivity check failed")
        result["neo4j"] = False

    # pgvector
    try:
        conn = await asyncpg.connect(_pg_dsn())
        await conn.close()
        result["pgvector"] = True
    except Exception:
        logger.exception("pgvector connectivity check failed")
        result["pgvector"] = False

    try:
        v = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd="/root/graphknows", text=True, timeout=3
        ).strip()
    except Exception:
        v = "unknown"

    result["version"] = v
    result["pid"] = os.getpid()

    return result


@app.tool()
async def memory_stats(session_id: str = "") -> dict[str, Any]:
    """Return graphknows memory statistics (STM and LTM counts).

    Args:
        session_id: Optional session to filter STM stats by.  Empty string
            returns aggregate counts.

    Returns:
        Dict with ``stm`` and ``ltm`` sub-dicts containing row counts.
    """
    import asyncpg

    async def _pg_counts() -> dict[str, int]:
        conn = await asyncpg.connect(_pg_dsn())
        try:
            sc = await conn.fetchval("SELECT count(*) FROM stm_chunks")
            st = await conn.fetchval("SELECT count(*) FROM stm_turns")
            lc = await conn.fetchval("SELECT count(*) FROM ltm_chunks")
            lt = await conn.fetchval("SELECT count(*) FROM ltm_turns")
            top = await conn.fetchval("SELECT count(*) FROM ltm_topics")
            return {
                "stm_chunks": int(sc),
                "stm_turns": int(st),
                "ltm_chunks": int(lc),
                "ltm_turns": int(lt),
                "ltm_topics": int(top),
            }
        finally:
            await conn.close()

    pg = await _pg_counts()

    from neo4j import GraphDatabase

    uri, user, pw = _neo4j_creds()
    drv = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    with drv.session() as s:
        ec = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()["c"]
        tc = s.run("MATCH (t:Topic) RETURN count(t) as c").single()["c"]
        sess_c = s.run("MATCH (s:SESSION) RETURN count(s) as c").single()["c"]
    drv.close()

    return {
        "stm": {
            "chunks": pg["stm_chunks"],
            "turns": pg["stm_turns"],
        },
        "ltm": {
            "sessions": int(sess_c),
            "entities": int(ec),
            "topics": int(tc),
            "chunks": pg["ltm_chunks"],
            "turns": pg["ltm_turns"],
        },
    }


@app.tool()
async def memory_ingest(
    text: str,
    session_id: str,
    title: str = "",
) -> dict[str, Any]:
    """Ingest text into graphknows short-term memory (Memgraph + pgvector STM).

    GLiNER2 entity extraction runs on the text; extracted entities are stored
    as nodes in Memgraph.  The text is chunked and embedded into pgvector stm_*.

    Note: First call takes ~15s while GLiNER2 model loads.

    Args:
        text: The text content to ingest.
        session_id: Session namespace for the STM graph.
        title: Optional document title for provenance.

    Returns:
        Dict with file_id, chunks, entities, elapsed_s, errors.
    """
    from graphknows.memory.extractor import GLiNER2EntityExtractor
    from graphknows.memory.ingestor import MemoryIngestor, _run
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())
    _run(pg.connect())

    extractor = GLiNER2EntityExtractor()
    rel_extractor = None
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            from graphknows.ontology.rel_extractor import OntologyRelExtractor

            rel_extractor = OntologyRelExtractor()
        except Exception:
            logger.exception("Failed to initialize OntologyRelExtractor")
    ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=_embed_fn(), rel_extractor=rel_extractor)
    result = ingestor.ingest_text(text, session_id=session_id, title=title)

    return {
        "file_id": result.file_id,
        "chunks": result.chunk_count,
        "entities": result.entity_count,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
    }


def _normalise_and_weight(hits: list, weight: float) -> None:
    """Min-max normalise scores in-place, then apply weight."""
    if not hits:
        return
    scores = [h.score for h in hits]
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    for h in hits:
        h.score = weight * ((h.score - mn) / rng if rng else 1.0)


@app.tool()
async def memory_recall(
    query: str,
    session_id: str,
    top_k: int = 5,
    scope: str = "stm",
) -> dict[str, Any]:
    """Recall from graphknows memory using hybrid RRF search.

    Args:
        query: Natural language query to recall against.
        session_id: Session namespace to scope the search.
        top_k: Maximum number of hits to return (default 5).
        scope: ``"stm"`` for short-term, ``"ltm"`` for long-term,
            ``"both"`` for RRF-merged results from both.

    Returns:
        Dict with ``hits`` list, each item having score, type, text,
        session_id, and entities fields.
    """
    from graphknows.memory.ingestor import _run
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    embed = _embed_fn()

    def _clean(text: str) -> str:
        r"""Strip the '## Content\n' heading artifact inserted by the formatter."""
        import re

        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _fmt(results: list) -> list[dict]:
        return [
            {
                "score": round(r.score, 6),
                "cosine_score": round(getattr(r, "cosine_score", r.score), 6),
                "type": r.source_type,
                "text": _clean(r.text),
                "session_id": r.provenance.get("session_id", session_id),
                "entities": list(r.entity_names or []),
            }
            for r in results
        ]

    if scope in ("stm", "both"):
        from graphknows.retrieval.recall_stm import recall_short_term

        mg = MemgraphStore(_mg_uri())
        pg = PgVectorStore(_pg_dsn())
        _run(pg.connect())
        stm_hits = recall_short_term(
            mg, pg, query, session_id=session_id, top_k=top_k, embed_fn=embed
        )
    else:
        stm_hits = []

    if scope in ("ltm", "both"):
        from graphknows.retrieval.recall_ltm import recall_long_term
        from graphknows.retrieval.stores.pgvector import PgVectorStore

        pg2 = PgVectorStore(_pg_dsn())
        _run(pg2.connect())
        neo = _get_neo()
        ltm_hits = recall_long_term(
            neo, pg2, query, top_k=top_k, session_filter=None, embed_fn=embed
        )
        neo.close()
    else:
        ltm_hits = []

    # Normalise cross-scope scores when merging both
    if scope in ("both",):
        _normalise_and_weight(stm_hits, weight=0.6)
        _normalise_and_weight(ltm_hits, weight=0.4)

    # Merge and re-rank by score if both scopes used
    all_hits = stm_hits + ltm_hits
    all_hits.sort(key=lambda r: r.score, reverse=True)

    return {"hits": _fmt(all_hits[:top_k])}


@app.tool()
async def memory_flush(session_id: str, run_gds: bool = False) -> dict[str, Any]:
    """Flush a session from short-term memory into long-term memory.

    Promotes Memgraph STM nodes → Neo4j LTM with entity merging.
    Louvain topic summarization and GDS enrichment are skipped by default;
    set ``run_gds=True`` for full enrichment (Node2Vec, Leiden, PageRank, etc.).

    Args:
        session_id: Session to promote from STM to LTM.
        run_gds: Whether to run GDS enrichment after the flush.

    Returns:
        Dict with nodes, edges, chunks, turns, elapsed_s, errors.
    """
    from graphknows.memory.flush import FlushOptions, SessionFlusher
    from graphknows.retrieval.stores.memgraph import MemgraphStore
    from graphknows.retrieval.stores.neo4j import Neo4jStore
    from graphknows.retrieval.stores.pgvector import PgVectorStore

    mg = MemgraphStore(_mg_uri())
    pg = PgVectorStore(_pg_dsn())
    await pg.connect()

    uri, user, pw = _neo4j_creds()
    neo = Neo4jStore(uri=uri, user=user, password=pw)

    key = os.environ.get("OPENROUTER_API_KEY", "")
    flusher = SessionFlusher(mg, pg, neo, openrouter_key=key)
    opts = FlushOptions(
        summarize_topics=False,
        run_gds=run_gds,
        clear_stm=True,
        merge_entities=True,
    )

    result = await flusher.flush(session_id, opts)

    return {
        "nodes": result.nodes_written,
        "edges": result.edges_written,
        "chunks": result.chunks_promoted,
        "turns": result.turns_promoted,
        "elapsed_s": round(result.elapsed_s, 3),
        "errors": list(result.errors),
    }


# ---------------------------------------------------------------------------
# LTM tools — merged from graphknows/mcp_ltm.py
# ---------------------------------------------------------------------------


@app.tool()
async def ltm_query(query: str, top_k: int = 5) -> dict[str, Any]:
    """Semantic search over LTM topic summaries.

    Embeds the query and returns the nearest matching topics from the
    pgvector ltm_topics table.

    Args:
        query: Natural-language search query.
        top_k: Number of topics to return (1–20, default 5).

    Returns:
        Dict with a ``results`` key containing a list of matching topics.
    """
    import json

    import asyncpg as apg

    from graphknows.retrieval.stores.embedder import embed_one

    top_k = max(1, min(top_k, 20))
    embedding = json.dumps(embed_one(query))

    conn = await apg.connect(_pg_dsn())
    try:
        rows = await conn.fetch(
            "SELECT topic_id, summary, entity_names, session_ids FROM ltm_topics "
            f"ORDER BY embedding <=> $1::vector LIMIT {top_k}",
            embedding,
        )
    finally:
        await conn.close()

    results = []
    for r in rows:
        import json as _json

        entity_names = r["entity_names"]
        if isinstance(entity_names, str):
            entity_names = _json.loads(entity_names)
        session_ids = r["session_ids"]
        if isinstance(session_ids, str):
            session_ids = _json.loads(session_ids)
        results.append(
            {
                "topic_id": r["topic_id"],
                "summary": r["summary"],
                "entity_names": entity_names or [],
                "session_ids": session_ids or [],
            }
        )
    return {"results": results}


@app.tool()
def ltm_entity(name: str) -> dict[str, Any]:
    """Look up a named entity in the LTM Neo4j graph.

    Returns all properties including community, PageRank, betweenness,
    first/last seen timestamps, and segment count.

    Args:
        name: Entity name (case-sensitive).

    Returns:
        Dict with entity properties, or ``{"error": "..."}`` if not found.
    """
    from neo4j import GraphDatabase

    uri, user, pw = _neo4j_creds()
    d = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    try:
        with d.session() as s:
            row = s.run(
                "MATCH (e:Entity {name: $name}) "
                "OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:Segment) "
                "RETURN properties(e) AS props, count(DISTINCT seg) AS segment_count",
                name=name,
            ).single()
    finally:
        d.close()

    if row is None:
        return {"error": f"Entity '{name}' not found"}

    props = dict(row["props"])
    props["segment_count"] = row["segment_count"]
    return props


@app.tool()
def ltm_topics(min_entities: int = 3) -> dict[str, Any]:
    """List all Topic nodes with an entity count above the threshold.

    Ordered by entity_count descending.

    Args:
        min_entities: Minimum entity count threshold (default 3).

    Returns:
        Dict with a ``topics`` key containing the topic list.
    """
    from neo4j import GraphDatabase

    uri, user, pw = _neo4j_creds()
    d = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    try:
        with d.session() as s:
            rows = s.run(
                "MATCH (t:Topic) "
                "WHERE t.entity_count >= $min_entities "
                "RETURN t.community_id AS community_id, "
                "       t.summary AS summary, "
                "       t.entity_count AS entity_count "
                "ORDER BY t.entity_count DESC",
                min_entities=min_entities,
            ).data()
    finally:
        d.close()

    return {"topics": rows}


@app.tool()
def recall_by_document(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find document chunks whose mentioned entities match the query.

    Returns chunks ranked by number of matched entities.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1–20, default 5).

    Returns:
        Dict with a ``results`` key containing matched chunks.
    """
    from neo4j import GraphDatabase

    from graphknows.recall import recall_by_document as _rbd

    top_k = max(1, min(top_k, 20))
    uri, user, pw = _neo4j_creds()
    d = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    try:
        results = _rbd(d, query, top_k)
    finally:
        d.close()

    return {
        "results": [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ]
    }


@app.tool()
def recall_by_dialog(
    query: str,
    top_k: int = 5,
    session_filter: str = "",
) -> dict[str, Any]:
    """Find past conversation utterances whose mentioned entities match.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1–20, default 5).
        session_filter: Optional session_id to scope results.

    Returns:
        Dict with a ``results`` key containing matched utterances.
    """
    from neo4j import GraphDatabase

    from graphknows.recall import recall_by_dialog as _rbdia

    top_k = max(1, min(top_k, 20))
    filt = session_filter or None
    uri, user, pw = _neo4j_creds()
    d = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    try:
        results = _rbdia(d, query, top_k, filt)
    finally:
        d.close()

    return {
        "results": [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ]
    }


@app.tool()
def recall_by_topic(query: str, top_k: int = 5) -> dict[str, Any]:
    """Find topics whose label or member entities match the query.

    Args:
        query: Free-text search query.
        top_k: Max results to return (1–20, default 5).

    Returns:
        Dict with a ``results`` key containing matched topics.
    """
    from neo4j import GraphDatabase

    from graphknows.recall import recall_by_topic as _rbt

    top_k = max(1, min(top_k, 20))
    uri, user, pw = _neo4j_creds()
    d = GraphDatabase.driver(uri, auth=(user, pw))  # NOSONAR
    try:
        results = _rbt(d, query, top_k)
    finally:
        d.close()

    return {
        "results": [
            {
                "text": r.text,
                "entities": r.entity_names,
                "provenance": r.provenance,
                "score": r.score,
            }
            for r in results
        ]
    }


@app.tool()
async def corpus_ingest(
    documents: list[dict],
    taxonomy_file: str | None = None,
) -> dict:
    """Ingest a corpus of documents into the long-term knowledge graph.

    Runs the full graphgen pipeline: NER → relation extraction → graph
    cleaning → community detection → Neo4j write-back.  Use for batch
    knowledge ingestion.  GDS enrichment (Node2Vec + Louvain) runs
    automatically after write.

    Args:
        documents: List of document dicts. Each must have:
            - text (str): document content
            - title (str): human-readable title
            - id (str): unique document identifier
            - source (str, optional): provenance URL or label
        taxonomy_file: Optional path or name of taxonomy YAML to use.
            Defaults to taxonomy.generic.yaml.  Pass
            "taxonomy.eu-politics.yaml" for EU political domain.

    Returns:
        Dict with pipeline statistics: documents_processed,
        entities_extracted, relations_extracted, nodes_written,
        edges_written, errors (list).
    """
    if not documents:
        return {"documents_processed": 0, "error": "no documents provided"}
    try:
        from graphknows.graphgen.ingest import run_ingestion

        config: dict = {}
        if taxonomy_file:
            config["taxonomy_file"] = taxonomy_file
        stats = await run_ingestion(documents=documents, config_override=config or None)
        return stats
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "documents_processed": 0}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the graphknows MCP stdio server.

    Reads JSON-RPC 2.0 requests on stdin, writes responses on stdout.
    All logging goes to stderr.
    """
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
