#!/usr/bin/env python3
"""Multi-source EU document ingest using all 9 BaseSource adapters.

Fetches 2 documents from each adapter, runs them through:
  1. markdown_to_chunks()  → chunk dicts
  2. Embedding via BGE-small-en-v1.5 → pgvector chunks table
  3. Document node upsert in Neo4j with FROM edges to Chunks
  4. CO_OCCURS materialisation for new entities (optional, no GLiNER2 needed)

Usage::

    PYTHONPATH=/root/graphknows python3 scripts/multi_source_ingest.py [--limit 2] [--dry-run]

Notes:
  - Adapters that inherit BaseSource expose ``scrape(limit=N)`` and ``fetch_one(url)``.
  - Network failures are logged and skipped; the script never hard-crashes on a single
    adapter failure.
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

# ── Adapter registry ─────────────────────────────────────────────────────────
# Only BaseSource-derived adapters (they have scrape()); SourceAdapter-based ones
# (ep_debates, eurlex, uk_legislation) need their scrapers and are handled separately.

def _build_base_source_adapters() -> list:
    """Instantiate all BaseSource adapters (rate-limited HTTP scrapers)."""
    from services.search.adapters.council_conclusions import CouncilConclusionsSource
    from services.search.adapters.ec_press import ECPressSource
    from services.search.adapters.eca_reports import ECAReportsSource
    from services.search.adapters.ecb_publications import ECBPublicationsSource
    from services.search.adapters.ep_resolutions import EPResolutionsSource
    from services.search.adapters.eprs_briefings import EPRSBriefingsSource

    return [
        CouncilConclusionsSource(rate_limit_rps=0.5, timeout=60),
        ECPressSource(rate_limit_rps=1.0, timeout=45),
        ECAReportsSource(rate_limit_rps=0.5, timeout=60),
        ECBPublicationsSource(rate_limit_rps=0.5, timeout=60),
        EPResolutionsSource(rate_limit_rps=0.5, timeout=45),
        EPRSBriefingsSource(rate_limit_rps=0.5, timeout=45),
    ]


def _run_scraper_fn(fn, *args, timeout: int = 60, **kwargs) -> list:
    """Run a possibly-async scraper function with a timeout.

    Args:
        fn: Function to call (sync or async).
        *args: Positional arguments forwarded to ``fn``.
        timeout: Max seconds to wait (default 60).
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Return value of ``fn``, or empty list on error/timeout.
    """
    import inspect

    if inspect.iscoroutinefunction(fn):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
            )
        except (asyncio.TimeoutError, Exception) as exc:
            log.warning("Async scraper timed out or failed: %s", exc)
            return []
        finally:
            loop.close()
    else:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.warning("Sync scraper failed: %s", exc)
            return []


def _scrape_eurlex(limit: int) -> list[ParsedDocument]:
    """Fetch EUR-Lex docs via its scraper module directly."""
    try:
        import scrapers.eurlex as _eurlex  # type: ignore[import]

        if not hasattr(_eurlex, "run_incremental"):
            log.warning("[eurlex] run_incremental not found")
            return []

        raw_docs = _run_scraper_fn(_eurlex.run_incremental, days=3, timeout=45)
        raw_docs = raw_docs[:limit]
    except Exception as exc:
        log.warning("[eurlex] scraper failed: %s", exc)
        return []

    docs = []
    for d in raw_docs:
        try:
            doc = ParsedDocument(
                source_id="eurlex",
                source_type="EU_LEGISLATION",
                url=d.get("url", d.get("source_url", "")),
                title=d.get("title", "EUR-Lex Document"),
                date=d.get("published", d.get("date", "2024-01-01"))[:10],
                institution="EUR-Lex",
                language="EN",
                raw_text=d.get("text", "")[:50_000],
                markdown=f"# {d.get('title', 'EUR-Lex Document')}\n\n{d.get('text', '')}",
                doc_id=d.get("celex", ""),
            )
            docs.append(doc)
        except Exception as exc:
            log.warning("[eurlex] doc parse error: %s", exc)
    return docs


def _scrape_ep_debates(limit: int) -> list[ParsedDocument]:
    """Fetch EP Debates via the underlying scraper module."""
    try:
        import scrapers.ep_debates as _ep  # type: ignore[import]

        if not hasattr(_ep, "run_incremental"):
            log.warning("[ep_debates] run_incremental not found")
            return []

        raw_docs = _run_scraper_fn(_ep.run_incremental, max_speeches=50, timeout=45)
        raw_docs = raw_docs[:limit]
    except Exception as exc:
        log.warning("[ep_debates] scraper failed: %s", exc)
        return []

    docs = []
    for d in raw_docs:
        try:
            text = d.get("text", d.get("content", ""))
            title = d.get("title", d.get("itm_title", "EP Debate"))
            doc = ParsedDocument(
                source_id="ep_debates",
                source_type="EP_DEBATE",
                url=d.get("url", "https://europarl.europa.eu/plenary/en/debates.html"),
                title=title,
                date=d.get("date", d.get("session_date", "2024-01-01"))[:10],
                institution="European Parliament",
                language="EN",
                raw_text=text[:50_000],
                markdown=f"# {title}\n\n{text}",
                doc_id=d.get("itm_id", d.get("id", "")),
            )
            docs.append(doc)
        except Exception as exc:
            log.warning("[ep_debates] doc parse error: %s", exc)
    return docs


def _scrape_uk_legislation(limit: int) -> list[ParsedDocument]:
    """Fetch UK legislation via its scraper module."""
    try:
        import scrapers.uk_legislation as _ukleg  # type: ignore[import]

        if not hasattr(_ukleg, "run_incremental"):
            log.warning("[uk_legislation] run_incremental not found")
            return []

        raw_docs = _run_scraper_fn(_ukleg.run_incremental, days=3, timeout=45)
        raw_docs = raw_docs[:limit]
    except Exception as exc:
        log.warning("[uk_legislation] scraper failed: %s", exc)
        return []

    docs = []
    for d in raw_docs:
        try:
            text = d.get("text", d.get("content", ""))
            title = d.get("title", "UK Legislation")
            doc = ParsedDocument(
                source_id="uk_legislation",
                source_type="UK_LEGISLATION",
                url=d.get("url", ""),
                title=title,
                date=d.get("date", "2024-01-01")[:10],
                institution="UK Parliament",
                language="EN",
                raw_text=text[:50_000],
                markdown=f"# {title}\n\n{text}",
                doc_id=d.get("id", ""),
            )
            docs.append(doc)
        except Exception as exc:
            log.warning("[uk_legislation] doc parse error: %s", exc)
    return docs


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

async def ingest_base_source_adapter(adapter, limit: int, dry_run: bool) -> list[ParsedDocument]:
    """Run scrape() on a BaseSource adapter and return ParsedDocuments.

    Args:
        adapter: A BaseSource instance.
        limit: Max documents to fetch.
        dry_run: Passed through for logging.

    Returns:
        List of successfully parsed documents.
    """
    sid = adapter.source_id
    log.info("[%s] Starting scrape (limit=%d) …", sid, limit)
    async with adapter:
        try:
            docs = await adapter.scrape(limit=limit)
        except Exception as exc:
            log.warning("[%s] scrape() raised: %s", sid, exc)
            docs = []
    log.info("[%s] Scraped %d documents", sid, len(docs))
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

async def _run_async(limit: int, dry_run: bool) -> None:
    """Async main: scrape BaseSource adapters concurrently, then process."""
    adapters = _build_base_source_adapters()
    tasks = [ingest_base_source_adapter(a, limit, dry_run) for a in adapters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_docs: list[ParsedDocument] = []
    for adapter, result in zip(adapters, results):
        if isinstance(result, Exception):
            log.warning("[%s] adapter error: %s", adapter.source_id, result)
        else:
            all_docs.extend(result)

    return all_docs


def run(limit: int = 2, dry_run: bool = False) -> None:
    """Full ingest: all 9 adapters → pgvector + Neo4j.

    Args:
        limit: Max documents to fetch per adapter.
        dry_run: If True, extract and log but do not write to DBs.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    grand_total: dict = {"docs": 0, "chunks": 0, "adapters": 0}

    try:
        # ── BaseSource adapters (async HTTP scrapers) ──────────────────────
        loop = asyncio.new_event_loop()
        base_docs = loop.run_until_complete(_run_async(limit, dry_run))
        loop.close()

        if base_docs:
            summary = process_documents(base_docs, driver, dry_run)
            grand_total["docs"] += summary["docs"]
            grand_total["chunks"] += summary["chunks"]
            grand_total["adapters"] += 1

        # ── Scraper-module adapters (eurlex, ep_debates, uk_legislation) ──
        scraper_fns = [
            ("eurlex", _scrape_eurlex),
            ("ep_debates", _scrape_ep_debates),
            ("uk_legislation", _scrape_uk_legislation),
        ]
        for name, fn in scraper_fns:
            log.info("[%s] Fetching via scraper module …", name)
            try:
                docs = fn(limit)
            except Exception as exc:
                log.warning("[%s] scraper module failed: %s", name, exc)
                docs = []

            if docs:
                summary = process_documents(docs, driver, dry_run)
                grand_total["docs"] += summary["docs"]
                grand_total["chunks"] += summary["chunks"]
                grand_total["adapters"] += 1
            else:
                log.info("[%s] No documents retrieved", name)

    finally:
        driver.close()

    log.info(
        "═══ Ingest complete. adapters_with_data=%d | docs=%d | chunks=%d%s ═══",
        grand_total["adapters"],
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
