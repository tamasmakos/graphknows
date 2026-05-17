#!/usr/bin/env python3
"""Multi-source EU document ingest using all search service connectors.

Fetches documents from each connector, runs them through:
  1. markdown_to_chunks()  → chunk dicts
  2. Embedding via BGE-small-en-v1.5 → pgvector chunks table
  3. Document node upsert in Neo4j with FROM edges to Chunks

Usage::

    PYTHONPATH=/root/graphknows python3 scripts/multi_source_ingest.py [--limit 2] [--dry-run]

Notes:
  - All connectors live in services/search/connectors/sources/.
  - Network failures are logged and skipped; the script never hard-crashes on a single
    connector failure.
  - Embedding requires sentence-transformers + BAAI/bge-small-en-v1.5 (already cached
    from the original ingest run).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psycopg2
from dotenv import dotenv_values
from neo4j import GraphDatabase

# ── Repo path setup ──────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "services"))

from services.search.models import SearchQuery
from services.search.parsed_document import ParsedDocument, markdown_to_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
log = logging.getLogger(__name__)

_ENV = dotenv_values(str(_REPO / ".env"))

# ── Connection config ────────────────────────────────────────────────────────
NEO4J_URI = f"bolt://{_ENV.get('NEO4J_HOST', '172.18.0.1')}:{_ENV.get('NEO4J_PORT', 7687)}"
NEO4J_AUTH = (_ENV.get("NEO4J_USER", "neo4j"), _ENV.get("NEO4J_PASSWORD", "testpassword123"))

_PG_DSN = (
    f"postgresql://{_ENV.get('PGVECTOR_USER', 'postgres')}"
    f":{_ENV.get('PGVECTOR_PASSWORD', 'pgvectorpass123')}"
    f"@{_ENV.get('PGVECTOR_HOST', '172.18.0.1')}"
    f":{_ENV.get('PGVECTOR_PORT', 5432)}"
    f"/{_ENV.get('PGVECTOR_DB', 'vectordb')}"
)

EMBED_MODEL = _ENV.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ── Connector registry ────────────────────────────────────────────────────────

def _build_connectors() -> list:
    """Instantiate all search service connectors."""
    from services.search.connectors.sources.council_conclusions import CouncilConclusionsConnector
    from services.search.connectors.sources.ec_press import ECPressConnector
    from services.search.connectors.sources.eca_reports import ECAReportsConnector
    from services.search.connectors.sources.ecb_publications import ECBPublicationsConnector
    from services.search.connectors.sources.ep_debates import EPDebatesConnector
    from services.search.connectors.sources.ep_resolutions import EPResolutionsConnector
    from services.search.connectors.sources.eprs_briefings import EPRSBriefingsConnector
    from services.search.connectors.sources.eurlex import EurLexConnector
    from services.search.connectors.sources.uk_legislation import UKLegislationConnector
    from services.search.connectors.sources.wikipedia_eu import WikipediaEUConnector
    from services.search.connectors.sources.cellar_sparql import CellarSPARQLConnector

    return [
        CouncilConclusionsConnector(rate_limit_rps=0.5, timeout=60),
        ECPressConnector(rate_limit_rps=1.0, timeout=45),
        ECAReportsConnector(rate_limit_rps=0.5, timeout=60),
        ECBPublicationsConnector(rate_limit_rps=0.5, timeout=60),
        EPDebatesConnector(),
        EPResolutionsConnector(rate_limit_rps=0.5, timeout=45),
        EPRSBriefingsConnector(rate_limit_rps=0.5, timeout=45),
        EurLexConnector(),
        UKLegislationConnector(),
        WikipediaEUConnector(rate_limit_rps=1.0, timeout=30),
        CellarSPARQLConnector(doc_type="press_release", rate_limit_rps=0.5, timeout=30),
        CellarSPARQLConnector(doc_type="regulation", rate_limit_rps=0.5, timeout=30),
    ]


def _to_parsed(result, doc) -> ParsedDocument:
    """Convert a (SearchResult, Document) pair to a ParsedDocument for ingestion."""
    text = doc.text if doc else ""
    title = result.title or "Untitled"
    return ParsedDocument(
        source_id=result.source,
        source_type=result.source.upper().replace("-", "_"),
        url=result.url or "",
        title=title,
        date=(result.date or "")[:10],
        institution=doc.metadata.get("institution", "") if doc else "",
        language=doc.metadata.get("language", "EN") if doc else "EN",
        raw_text=text[:50_000],
        markdown=f"# {title}\n\n{text}",
        doc_id=result.id,
    )


# ── Embedding ────────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    """Return singleton SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s …", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings with BGE-small-en-v1.5, L2-normalised."""
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── Storage helpers ──────────────────────────────────────────────────────────

def store_chunks_pg(chunks: list[dict], dry_run: bool = False) -> int:
    """Insert/upsert chunks (with embeddings) into pgvector.

    Args:
        chunks: Chunk dicts enriched with an ``embedding`` key (list[float]).
        dry_run: If True, skip actual DB writes.

    Returns:
        Number of chunks upserted.
    """
    if dry_run:
        return len(chunks)

    conn = psycopg2.connect(_PG_DSN)
    try:
        cur = conn.cursor()
        for c in chunks:
            cur.execute(
                """
                INSERT INTO chunks (id, doc_id, text, embedding, entity_count)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                  SET text=EXCLUDED.text, embedding=EXCLUDED.embedding,
                      entity_count=EXCLUDED.entity_count
                """,
                (
                    c["chunk_id"],
                    c["doc_id"],
                    c["text"][:4000],
                    c["embedding"],
                    0,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    return len(chunks)


def store_document_neo4j(session, doc: ParsedDocument, chunks: list[dict]) -> None:
    """Upsert a Document node and create FROM edges for each chunk.

    Args:
        session: Active Neo4j session.
        doc: ParsedDocument metadata.
        chunks: Chunk dicts associated with this document.
    """
    props = {
        "id": doc.doc_id,
        "title": doc.title,
        "institution": doc.institution,
        "source_id": doc.source_id,
        "source_type": doc.source_type,
        "date": doc.date,
        "url": doc.url,
        "language": doc.language,
    }
    session.run(
        "MERGE (d:Document {id: $id}) SET d += $props",
        id=doc.doc_id,
        props=props,
    )

    for c in chunks:
        session.run(
            """
            MERGE (ch:Chunk {id: $chunk_id})
            SET ch.doc_id = $doc_id,
                ch.text   = $text,
                ch.heading = $heading,
                ch.source_id = $source_id,
                ch.date    = $date
            WITH ch
            MATCH (d:Document {id: $doc_id})
            MERGE (ch)-[:FROM]->(d)
            """,
            chunk_id=c["chunk_id"],
            doc_id=doc.doc_id,
            text=c["text"][:2000],
            heading=c.get("heading", ""),
            source_id=doc.source_id,
            date=doc.date,
        )


# ── Ingest orchestration ─────────────────────────────────────────────────────

async def _ingest_connector(connector, limit: int, dry_run: bool) -> list[ParsedDocument]:
    """Search a connector for recent documents and return ParsedDocuments.

    Args:
        connector: A SourceConnector instance.
        limit: Max documents to fetch.
        dry_run: Passed through for logging.

    Returns:
        List of successfully parsed documents.
    """
    sid = connector.source_id
    log.info("[%s] Starting fetch (limit=%d) …", sid, limit)
    q = SearchQuery(query="", limit=limit)
    try:
        results = await connector.search(q)
    except Exception as exc:
        log.warning("[%s] search() raised: %s", sid, exc)
        return []

    docs: list[ParsedDocument] = []
    for result in results[:limit]:
        try:
            full_doc = await connector.fetch(result.id)
            if full_doc is None:
                log.warning("[%s] fetch() returned None for id=%s", sid, result.id)
                continue
            docs.append(_to_parsed(result, full_doc))
        except Exception as exc:
            log.warning("[%s] fetch() raised for id=%s: %s", sid, result.id, exc)

    log.info("[%s] Fetched %d documents", sid, len(docs))
    return docs


def process_documents(
    docs: list[ParsedDocument],
    driver,
    dry_run: bool,
) -> dict:
    """Chunk, embed and persist a list of ParsedDocuments.

    Args:
        docs: Parsed documents to ingest.
        driver: Active Neo4j driver.
        dry_run: If True, skip actual DB writes.

    Returns:
        Summary dict with counts.
    """
    total_chunks = 0
    total_docs = 0

    for doc in docs:
        if not doc.markdown.strip():
            log.warning("Skipping document %s — empty markdown", doc.doc_id)
            continue

        chunks = markdown_to_chunks(doc)
        if not chunks:
            log.warning("No chunks for document %s", doc.doc_id)
            continue

        # Embed
        texts = [c["text"] for c in chunks]
        try:
            embeddings = embed_texts(texts)
        except Exception as exc:
            log.warning("Embedding failed for doc %s: %s", doc.doc_id, exc)
            continue

        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb.tolist()

        # pgvector
        n_pg = store_chunks_pg(chunks, dry_run=dry_run)

        # Neo4j
        if not dry_run:
            with driver.session() as session:
                store_document_neo4j(session, doc, chunks)
        else:
            log.info(
                "[DRY] Would store doc=%s, %d chunks → pgvector + Neo4j",
                doc.doc_id,
                len(chunks),
            )

        total_chunks += n_pg
        total_docs += 1
        log.info(
            "Stored doc=%s (%s) | %d chunks | title=%s",
            doc.doc_id,
            doc.source_id,
            len(chunks),
            doc.title[:80],
        )

    return {"docs": total_docs, "chunks": total_chunks}


# ── Main ─────────────────────────────────────────────────────────────────────

async def _run_async(limit: int, dry_run: bool) -> list[ParsedDocument]:
    """Async main: fetch all connectors concurrently, then return docs."""
    connectors = _build_connectors()
    tasks = [_ingest_connector(c, limit, dry_run) for c in connectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_docs: list[ParsedDocument] = []
    for connector, result in zip(connectors, results):
        if isinstance(result, Exception):
            log.warning("[%s] connector error: %s", connector.source_id, result)
        else:
            all_docs.extend(result)

    return all_docs


def run(limit: int = 2, dry_run: bool = False) -> None:
    """Full ingest: all connectors → pgvector + Neo4j.

    Args:
        limit: Max documents to fetch per connector.
        dry_run: If True, extract and log but do not write to DBs.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    grand_total: dict = {"docs": 0, "chunks": 0, "connectors": 0}

    try:
        # ── All connectors (async HTTP) ────────────────────────────────────
        loop = asyncio.new_event_loop()
        all_docs = loop.run_until_complete(_run_async(limit, dry_run))
        loop.close()

        if all_docs:
            summary = process_documents(all_docs, driver, dry_run)
            grand_total["docs"] += summary["docs"]
            grand_total["chunks"] += summary["chunks"]
            grand_total["connectors"] += 1

    finally:
        driver.close()

    log.info(
        "═══ Ingest complete. connectors_with_data=%d | docs=%d | chunks=%d%s ═══",
        grand_total["connectors"],
        grand_total["docs"],
        grand_total["chunks"],
        " (dry-run)" if dry_run else "",
    )

    # Final Neo4j summary
    if not dry_run:
        _verify_neo4j()


def _verify_neo4j() -> None:
    """Log current node/edge counts from Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        with driver.session() as session:
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
            ).data()
            log.info("Neo4j node counts: %s", counts)
            from_edges = session.run("MATCH ()-[r:FROM]->() RETURN count(r) AS n").single()["n"]
            log.info("Neo4j FROM edges: %d", from_edges)
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source EU document ingest")
    parser.add_argument(
        "--limit", type=int, default=2, help="Max documents per adapter (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch and log but do not write to DBs"
    )
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run)
