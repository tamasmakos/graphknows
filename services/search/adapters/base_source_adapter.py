"""Bridge: wraps a BaseSource into the SourceAdapter interface.

BaseSource (scrape / fetch_one) and SourceAdapter (search / fetch / canonical_id)
are two different ABC families that grew up side by side.  This adapter lets
BaseSource implementations participate in the unified search router without
requiring them to be rewritten.

Usage (in router.py):
    from .adapters.base_source_adapter import wrap
    from .adapters.ep_resolutions import EPResolutionsSource

    _ADAPTERS["ep_resolutions"] = wrap(EPResolutionsSource())
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument
from ..models import Document, SearchQuery, SearchResult
from .base import SourceAdapter

log = logging.getLogger(__name__)


class BaseSourceAdapter(SourceAdapter):
    """SourceAdapter wrapper around any BaseSource implementation.

    Maps:
        SourceAdapter.search()       → BaseSource.scrape() + token filter + doc cache
        SourceAdapter.fetch()        → cache hit first, then fetch_one(url) fallback
        SourceAdapter.canonical_id() → ParsedDocument.doc_id

    The adapter keeps a shared doc cache (_doc_cache) so that fetch() can
    return a document that was already retrieved during search() without
    triggering a second HTTP round-trip.  This avoids the O(n_docs × batch)
    re-scrape that the naive re-scrape approach caused.
    """

    def __init__(self, inner: BaseSource) -> None:
        self._inner = inner
        self.source_id: str = inner.source_id
        self.rate_limit: float = getattr(inner, "_rate_limit_rps", 1.0)
        # doc_id → ParsedDocument; populated during search(), consumed by fetch()
        self._doc_cache: dict[str, ParsedDocument] = {}

    # ------------------------------------------------------------------ #
    # SourceAdapter interface                                               #
    # ------------------------------------------------------------------ #

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        limit = query.limit or 10
        try:
            async with self._inner:
                docs = await self._inner.scrape(limit=limit * 2)
        except Exception as exc:
            log.warning("[%s] scrape failed: %s", self.source_id, exc)
            return []

        q_tokens = set(query.query.lower().split())
        # Build word-boundary patterns to avoid "euro" matching "European"
        q_patterns = [re.compile(r'\b' + re.escape(tok) + r'\b') for tok in q_tokens]
        results: list[SearchResult] = []
        for doc in docs:
            combined = (
                (doc.title or "") + " " + (doc.raw_text or "") + " " + (doc.markdown or "")
            ).lower()
            if not any(pat.search(combined) for pat in q_patterns):
                continue
            if query.date_from and doc.date and doc.date < query.date_from:
                continue
            if query.date_to and doc.date and doc.date > query.date_to:
                continue

            # Cache so fetch() can serve this doc without re-scraping.
            self._doc_cache[doc.doc_id] = doc

            results.append(SearchResult(
                id=doc.doc_id,
                source=self.source_id,
                title=doc.title,
                date=doc.date or None,
                url=doc.url,
                excerpt=(doc.raw_text or doc.markdown or "")[:300] or None,
            ))
            if len(results) >= limit:
                break

        return results

    async def fetch(self, document_id: str) -> Document:
        # Fast path: doc was already retrieved during search().
        cached = self._doc_cache.get(document_id)
        if cached is not None:
            return self._parsed_to_document(cached)

        # Slow path: try fetch_one(url) if the source supports it.
        # We cannot reverse a SHA1 doc_id to a URL, so this only works when
        # the source exposes fetch_one() and we have a URL stored somewhere.
        # For sources that don't, raise cleanly rather than re-scraping.
        fetch_one = getattr(self._inner, "fetch_one", None)
        if fetch_one is not None:
            try:
                async with self._inner:
                    doc = await fetch_one(document_id)  # doc_id may be a URL for some sources
                if doc is not None:
                    return self._parsed_to_document(doc)
            except Exception as exc:
                log.warning("[%s] fetch_one failed for %s: %s", self.source_id, document_id, exc)

        raise FileNotFoundError(f"{self.source_id} document not found: {document_id}")

    def _parsed_to_document(self, doc: ParsedDocument) -> Document:
        return Document(
            id=doc.doc_id,
            source=self.source_id,
            title=doc.title,
            text=doc.raw_text or doc.markdown or "",
            metadata={
                "url": doc.url,
                "date": doc.date,
                "institution": doc.institution,
                "source_type": doc.source_type,
            },
            ingested_at=datetime.utcnow().isoformat(),
        )

    def canonical_id(self, raw: dict) -> str:
        url = raw.get("url", "")
        return hashlib.sha1(f"{self.source_id}:{url}".encode()).hexdigest()[:16]

    def canonical_id(self, raw: dict) -> str:
        url = raw.get("url", "")
        return hashlib.sha1(f"{self.source_id}:{url}".encode()).hexdigest()[:16]


def wrap(inner: BaseSource) -> BaseSourceAdapter:
    """Convenience factory — wrap a BaseSource into a SourceAdapter."""
    return BaseSourceAdapter(inner)
