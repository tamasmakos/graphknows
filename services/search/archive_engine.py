"""Archive Engine — on-demand fetch, light rerank, and Memgraph doc cache.

This module implements the search_archives agent tool:

  1. Agent emits an ArchiveQuery (structured DSL: archive ids + filters)
  2. Engine fans out to selected adapters in parallel
  3. Fetched docs are lightly reranked by cosine similarity against the query
  4. Results are MERGE-d into Memgraph as :CorpusDocument nodes (idempotent cache)
  5. Top-K docs returned to agent context window

No pre-indexing. No batch pipeline. Query-time only.
Documents accumulate in Memgraph as a natural cache — repeat queries are free.

Integration point:
    from search.archive_engine import ArchiveEngine, ArchiveQuery
    engine = ArchiveEngine(ws_broadcaster, session_id)
    results = await engine.search(query)
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

import numpy as np

log = logging.getLogger("search.archive_engine")

# ── Query DSL ─────────────────────────────────────────────────────────────────

@dataclass
class ArchiveQuery:
    """Structured archive query emitted by the agent's archive_selector.

    This is the DSL — the agent fills this in; the engine executes it.

    Attributes:
        query: Natural language search query.
        archives: List of archive IDs from REGISTRY (e.g. ["eurlex", "ep_debates"]).
                  If empty, engine selects all archives (not recommended).
        date_from: ISO date filter YYYY-MM-DD (optional).
        date_to: ISO date filter YYYY-MM-DD (optional).
        limit_per_archive: Max docs to fetch from each archive (default 5).
        top_k: Final number of results to return after reranking (default 8).
    """
    query: str
    archives: list[str]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit_per_archive: int = 5
    top_k: int = 8


@dataclass
class ArchiveResult:
    """Single retrieved and reranked document.

    Attributes:
        doc_id: Canonical document identifier (CELEX, URL, etc).
        title: Document title.
        excerpt: First 400 chars of text.
        url: Source URL.
        source: Archive id (e.g. "eurlex").
        date: Publication date string.
        score: Cosine similarity score against the query (0–1).
        from_cache: True if the doc was already in Memgraph (not re-fetched).
    """
    doc_id: str
    title: str
    excerpt: str
    url: str
    source: str
    date: str
    score: float
    from_cache: bool = False


# ── Engine ────────────────────────────────────────────────────────────────────

class ArchiveEngine:
    """On-demand archive fetcher with light cosine reranking and Memgraph caching.

    Args:
        memgraph_uri: Bolt URI for Memgraph (default from env or localhost).
        embed_model_name: SentenceTransformer model name.
    """

    _embed_model: Any = None  # module-level singleton, loaded lazily

    def __init__(
        self,
        memgraph_uri: str = "bolt://172.18.0.1:17687",
        embed_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._memgraph_uri = memgraph_uri
        self._embed_model_name = embed_model_name
        self._driver: Any = None

        # Ensure services/ is importable
        _svc = Path(__file__).resolve().parents[1]
        if str(_svc) not in sys.path:
            sys.path.insert(0, str(_svc))

    # ── Lazy singletons ───────────────────────────────────────────────────────

    def _get_model(self) -> Any:
        if ArchiveEngine._embed_model is None:
            from sentence_transformers import SentenceTransformer
            ArchiveEngine._embed_model = SentenceTransformer(self._embed_model_name)
            log.info("Loaded embedding model: %s", self._embed_model_name)
        return ArchiveEngine._embed_model

    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._memgraph_uri, auth=None)
        return self._driver

    # ── Cache: Memgraph ───────────────────────────────────────────────────────

    def _cache_lookup(self, doc_ids: list[str]) -> set[str]:
        """Return set of doc_ids already in Memgraph CorpusDocument cache."""
        if not doc_ids:
            return set()
        try:
            driver = self._get_driver()
            with driver.session() as s:
                rows = s.run(
                    "MATCH (n:CorpusDocument) WHERE n.doc_id IN $ids RETURN n.doc_id AS id",
                    ids=doc_ids,
                ).data()
            return {r["id"] for r in rows}
        except Exception as exc:
            log.warning("Memgraph cache lookup failed: %s", exc)
            return set()

    def _cache_store(self, docs: list[dict]) -> None:
        """MERGE fetched docs into Memgraph as :CorpusDocument nodes (idempotent)."""
        if not docs:
            return
        try:
            driver = self._get_driver()
            with driver.session() as s:
                s.run(
                    """
                    UNWIND $docs AS d
                    MERGE (n:CorpusDocument {doc_id: d.doc_id})
                    SET n.title   = d.title,
                        n.excerpt = d.excerpt,
                        n.url     = d.url,
                        n.source  = d.source,
                        n.date    = d.date
                    """,
                    docs=[
                        {
                            "doc_id": d["doc_id"],
                            "title":  d["title"][:300],
                            "excerpt": d["excerpt"][:500],
                            "url":    d["url"],
                            "source": d["source"],
                            "date":   d["date"],
                        }
                        for d in docs
                    ],
                )
            log.debug("Cached %d docs to Memgraph", len(docs))
        except Exception as exc:
            log.warning("Memgraph cache write failed: %s", exc)

    # ── Adapter fan-out ───────────────────────────────────────────────────────

    def _build_adapter_registry(self) -> dict[str, Any]:
        """Dynamically import all known adapters. Gracefully skips broken ones."""
        registry: dict[str, Any] = {}
        candidates = [
            ("search.adapters.ep_debates",          "EPDebatesAdapter",         "ep_debates"),
            ("search.adapters.eurlex",               "EurLexAdapter",            "eurlex"),
            ("search.adapters.ec_press",             "ECPressSource",            "ec_press"),
            ("search.adapters.ep_resolutions",       "EPResolutionsSource",      "ep_resolutions"),
            ("search.adapters.council_conclusions",  "CouncilConclusionsSource", "council_conclusions"),
            ("search.adapters.ecb_publications",     "ECBPublicationsSource",    "ecb_publications"),
            ("search.adapters.eprs_briefings",       "EPRSBriefingsSource",      "eprs_briefings"),
            ("search.adapters.uk_legislation",       "UKLegislationAdapter",     "uk_legislation"),
            ("search.adapters.wikipedia_eu",         "WikipediaEUAdapter",       "wikipedia_eu"),
            ("search.adapters.cellar_sparql",        "CellarSPARQLAdapter",      "cellar_sparql"),
        ]
        for mod_name, cls_name, key in candidates:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                registry[key] = getattr(mod, cls_name)
            except Exception:
                pass
        return registry

    async def _fetch_from_adapter(
        self,
        archive_id: str,
        adapter_cls: Any,
        aq: ArchiveQuery,
        cached_ids: set[str],
    ) -> list[dict]:
        """Fetch docs from one adapter, skip docs already cached."""
        from search.models import SearchQuery
        from search.adapters.base import SourceAdapter

        sq = SearchQuery(
            query=aq.query,
            limit=aq.limit_per_archive,
            date_from=aq.date_from,
            date_to=aq.date_to,
        )

        try:
            adapter = adapter_cls()

            if isinstance(adapter, SourceAdapter):
                results = await asyncio.wait_for(adapter.search(sq), timeout=15)
                docs = []
                for r in results[: aq.limit_per_archive]:
                    if r.id in cached_ids:
                        docs.append({
                            "doc_id":  r.id,
                            "title":   r.title,
                            "excerpt": r.excerpt or "",
                            "url":     r.url or "",
                            "source":  archive_id,
                            "date":    r.date or "",
                            "text":    r.excerpt or "",
                            "_cached": True,
                        })
                        continue
                    try:
                        doc = await asyncio.wait_for(adapter.fetch(r.id), timeout=10)
                        docs.append({
                            "doc_id":  r.id,
                            "title":   doc.title,
                            "excerpt": (doc.text or "")[:400],
                            "url":     r.url or doc.metadata.get("url", ""),
                            "source":  archive_id,
                            "date":    r.date or doc.metadata.get("published", ""),
                            "text":    (doc.text or "")[:2000],
                            "_cached": False,
                        })
                    except Exception as exc:
                        log.debug("fetch failed %s/%s: %s", archive_id, r.id, exc)
                return docs

            else:
                # Legacy adapter (search_query / scrape interface)
                async with adapter:
                    # Prefer search_query (keyword-aware) over scrape (recency-only)
                    search_fn = getattr(adapter, "search_query", None)
                    scrape_fn = getattr(adapter, "scrape", None)
                    if not callable(search_fn) and not callable(scrape_fn):
                        return []
                    if callable(search_fn):
                        raw = await asyncio.wait_for(
                            search_fn(aq.query, limit=aq.limit_per_archive), timeout=15
                        )
                    else:
                        # Legacy adapters: scrape(limit) only — no query arg
                        raw = await asyncio.wait_for(
                            scrape_fn(aq.limit_per_archive), timeout=15
                        )
                        # Post-filter by query token overlap
                        q_tokens = set(re.findall(r"\w+", aq.query.lower()))
                        filtered = [
                            d for d in raw
                            if q_tokens & set(re.findall(r"\w+",
                                ((getattr(d, "title", "") or "") + " " +
                                 str(getattr(d, "raw_text", "") or getattr(d, "markdown", "") or "")).lower()
                            ))
                        ]
                        raw = filtered or raw  # fall back to all if nothing matches
                    docs = []
                    for d in raw[: aq.limit_per_archive]:
                        text = str(getattr(d, "raw_text", None) or getattr(d, "markdown", None) or "")
                        doc_id = str(getattr(d, "doc_id", None) or getattr(d, "url", ""))
                        docs.append({
                            "doc_id":  doc_id,
                            "title":   getattr(d, "title", "") or "",
                            "excerpt": text[:400],
                            "url":     getattr(d, "url", "") or "",
                            "source":  archive_id,
                            "date":    str(getattr(d, "date", "") or ""),
                            "text":    text[:2000],
                            "_cached": doc_id in cached_ids,
                        })
                    return docs

        except asyncio.TimeoutError:
            log.warning("Adapter %s timed out", archive_id)
            return []
        except Exception as exc:
            log.warning("Adapter %s error: %s", archive_id, exc)
            return []

    # ── Light reranker ────────────────────────────────────────────────────────

    def _cosine_rerank(self, query: str, docs: list[dict], top_k: int) -> list[tuple[dict, float]]:
        """Embed query + doc texts, rerank by cosine similarity."""
        if not docs:
            return []
        model = self._get_model()
        texts = [f"{d['title']} {d['text']}" for d in docs]
        all_texts = [query] + texts
        embs = model.encode(all_texts, batch_size=32, normalize_embeddings=True)
        q_emb  = embs[0]
        d_embs = embs[1:]
        scores = d_embs @ q_emb  # cosine (normalized)
        ranked = sorted(zip(docs, scores.tolist()), key=lambda x: -x[1])
        return ranked[:top_k]

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(self, aq: ArchiveQuery) -> list[ArchiveResult]:
        """Execute an ArchiveQuery: fan-out → rerank → cache → return results.

        Args:
            aq: Structured archive query with archive IDs and filters.

        Returns:
            List of ArchiveResult sorted by relevance score, at most aq.top_k items.
        """
        if not aq.archives:
            log.warning("No archives specified in ArchiveQuery — returning empty")
            return []

        registry = self._build_adapter_registry()
        valid = [aid for aid in aq.archives if aid in registry]
        if not valid:
            log.warning("None of the requested archives are available: %s", aq.archives)
            return []

        # Parallel fan-out
        # First collect all possible doc IDs to do a single cache lookup
        tasks = [
            self._fetch_from_adapter(aid, registry[aid], aq, set())
            for aid in valid
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs: list[dict] = []
        for r in raw_results:
            if isinstance(r, list):
                all_docs.extend(r)

        if not all_docs:
            return []

        # Cache lookup (single round-trip)
        all_ids = [d["doc_id"] for d in all_docs if d["doc_id"]]
        cached_ids = self._cache_lookup(all_ids)
        for d in all_docs:
            d["_cached"] = d["doc_id"] in cached_ids

        # Light rerank
        ranked = self._cosine_rerank(aq.query, all_docs, top_k=aq.top_k)

        # Cache new docs (non-blocking best-effort)
        new_docs = [d for d, _ in ranked if not d.get("_cached")]
        if new_docs:
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    asyncio.to_thread(self._cache_store, new_docs)
                )
            )

        return [
            ArchiveResult(
                doc_id     = d["doc_id"],
                title      = d["title"],
                excerpt    = d["excerpt"],
                url        = d["url"],
                source     = d["source"],
                date       = d["date"],
                score      = round(float(score), 4),
                from_cache = d.get("_cached", False),
            )
            for d, score in ranked
        ]

    def format_for_agent(self, results: list[ArchiveResult]) -> str:
        """Format results as a compact string for the agent context window."""
        if not results:
            return "No results found in the selected archives."
        lines = []
        for i, r in enumerate(results, 1):
            cache_tag = " [cached]" if r.from_cache else ""
            lines.append(
                f"{i}. [{r.source}]{cache_tag} {r.title}\n"
                f"   Score: {r.score} | Date: {r.date} | {r.url}\n"
                f"   {r.excerpt[:300]}"
            )
        return "\n\n".join(lines)
