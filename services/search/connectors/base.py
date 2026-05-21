"""
Connector foundation: unified ABC for all EU data sources.

Class hierarchy
---------------
SourceConnector (ABC)          — every connector implements search() + fetch()
  ScrapingConnector            — for live-scraping connectors; implements
                                 search/fetch in terms of _scrape/_fetch_one
  HttpMixin                    — shared HTTP helpers (rate limiting, retries,
                                 HTML→Markdown, PDF→Markdown, lang detect)

Usage
-----
Disk-cached connectors (EurLex, EP Debates, UK Legislation) inherit
SourceConnector directly and implement search() + fetch() themselves.

Live scrapers (EP Resolutions, Council, Commission, ECB, ECA, EPRS,
Cellar SPARQL, Wikipedia, EP Open Data, data.europa.eu) inherit
ScrapingConnector and implement _scrape() (+ optionally _fetch_one()).
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import httpx

from ..models import Document, SearchQuery, SearchResult
from ..parsed_document import ParsedDocument

log = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; GraphKnows/1.0; +https://graphknows.eu)"
_DEFAULT_RETRIES = 3
_RETRY_BACKOFF = (1.0, 2.0, 4.0)


# ── Connector type taxonomy ───────────────────────────────────────────────────


class ConnectorType(str, enum.Enum):
    SPARQL = "sparql"
    REST_API = "rest_api"
    RSS_FEED = "rss_feed"
    HTML_SCRAPE = "html_scrape"
    DISK_CACHE = "disk_cache"


# ── Shared HTTP layer ─────────────────────────────────────────────────────────


class HttpMixin:
    """Rate-limited, retry-aware HTTP helpers shared by scraping connectors.

    Concrete classes must call _http_init() in __init__ before using any
    _get / _html_to_markdown / etc. methods.
    """

    _rate_limit_rps: float
    _timeout: int
    _semaphore: asyncio.Semaphore
    _min_interval: float
    _last_request: float
    _client: Optional[httpx.AsyncClient]

    def _http_init(self, rate_limit_rps: float = 1.0, timeout: int = 30) -> None:
        self._rate_limit_rps = rate_limit_rps
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(1)
        self._min_interval = 1.0 / max(rate_limit_rps, 0.1)
        self._last_request = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            )
        return self._client

    async def _get(self, url: str, **kwargs) -> str:
        """Rate-limited GET with exponential-backoff retries. Returns text."""
        client = self._ensure_client()
        async with self._semaphore:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

        last_exc: Exception = RuntimeError("unreachable")
        for attempt, backoff in enumerate(_RETRY_BACKOFF, 1):
            try:
                resp = await client.get(url, **kwargs)
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
            except httpx.RequestError as exc:
                last_exc = exc
            if attempt < _DEFAULT_RETRIES:
                await asyncio.sleep(backoff)

        raise last_exc

    def _html_to_markdown(self, html: str) -> str:
        # Strip nav/boilerplate tags and their content before conversion
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["nav", "footer", "header", "aside", "script", "style"]):
                tag.decompose()
            html = str(soup)
        except ImportError:
            html = re.sub(
                r"<(nav|footer|header|aside|script|style)[^>]*>.*?</\1>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
        try:
            import markdownify

            return markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()

    def _pdf_to_markdown(self, pdf_bytes: bytes) -> str:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = [page.get_text() for page in doc]
            return "\n\n".join(pages)
        except Exception as exc:
            log.warning("PDF extraction failed: %s", exc)
            return ""

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect

            return detect(text[:2000])
        except Exception:
            return "en"

    def _make_doc_id(self, *parts: str) -> str:
        raw = ":".join(parts)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def _extract_date(self, html: str, url: str = "") -> str | None:
        """Best-effort date extraction from HTML meta tags and URL patterns."""
        # Try structured meta
        for pat in (
            r'<meta[^>]*(?:name|property)=["\'](?:date|pubdate|article:published_time)["\'][^>]*content=["\']([^"\']+)',
            r'content=["\']([^"\']+)["\'][^>]*(?:name|property)=["\'](?:date|pubdate)',
            r'"datePublished"\s*:\s*"([^"]+)"',
        ):
            m = re.search(pat, html, re.IGNORECASE)
            if m:
                try:
                    return m.group(1)[:10]
                except Exception:
                    pass
        # Try URL year
        m = re.search(r"/(20\d{2})/", url)
        if m:
            return m.group(1)
        return None


# ── Unified abstract base ─────────────────────────────────────────────────────


class SourceConnector(ABC):
    """Every EU data connector implements this interface."""

    #: Unique identifier; must match source_manifest.yaml id
    source_id: str
    connector_type: ConnectorType
    rate_limit: float = 1.0

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Return ranked search results matching *query*."""

    @abstractmethod
    async def fetch(self, document_id: str) -> Document:
        """Return full document content for *document_id*."""

    def canonical_id(self, *parts: str) -> str:
        """Default: SHA-1 of ``source_id:part1:part2...`` (first 16 hex chars)."""
        raw = ":".join([self.source_id] + list(parts))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ── Scraping base (live HTTP connectors) ──────────────────────────────────────


class ScrapingConnector(SourceConnector, HttpMixin):
    """
    Implements search() and fetch() in terms of _scrape() and _fetch_one().

    Subclasses must implement _scrape(limit) → list[ParsedDocument].
    Optionally override _fetch_one(url) for document-level fetching.
    """

    connector_type: ConnectorType = ConnectorType.HTML_SCRAPE

    def __init__(self, rate_limit_rps: float = 1.0, timeout: int = 30) -> None:
        self._http_init(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._doc_cache: dict[str, ParsedDocument] = {}

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def _scrape(self, limit: int) -> list[ParsedDocument]:
        """Scrape up to *limit* documents and return ParsedDocument list."""

    async def _fetch_one(self, url: str) -> ParsedDocument | None:
        """Fetch a single document by URL. Override for connectors that support it."""
        return None

    # ── SourceConnector implementation ────────────────────────────────────────

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        self._ensure_client()
        try:
            docs = await self._scrape(query.limit * 2)
        except Exception as exc:
            log.warning("[%s] _scrape failed: %s", self.source_id, exc)
            return []

        q_tokens = set(query.query.lower().split())
        q_patterns = [re.compile(r"\b" + re.escape(t) + r"\b") for t in q_tokens]

        results: list[SearchResult] = []
        for doc in docs:
            combined = (
                (doc.title or "") + " " + (doc.raw_text or "") + " " + (doc.markdown or "")
            ).lower()
            if q_tokens and not any(pat.search(combined) for pat in q_patterns):
                continue
            if query.date_from and doc.date and doc.date < query.date_from:
                continue
            if query.date_to and doc.date and doc.date > query.date_to:
                continue

            self._doc_cache[doc.doc_id] = doc
            results.append(
                SearchResult(
                    id=doc.doc_id,
                    source=self.source_id,
                    title=doc.title,
                    date=doc.date or None,
                    url=doc.url,
                    excerpt=(doc.raw_text or doc.markdown or "")[:300] or None,
                )
            )
            if len(results) >= query.limit:
                break

        return results

    async def fetch(self, document_id: str) -> Document:
        cached = self._doc_cache.get(document_id)
        if cached is not None:
            return self._parsed_to_document(cached)
        self._ensure_client()
        # _fetch_one expects a URL; try to resolve from cache by doc_id
        doc = await self._fetch_one(document_id)
        if doc is not None:
            return self._parsed_to_document(doc)
        raise FileNotFoundError(f"[{self.source_id}] document not found: {document_id}")

    def canonical_id(self, *parts: str) -> str:
        raw = ":".join([self.source_id] + list(parts))
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    # ── Helper ────────────────────────────────────────────────────────────────

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
                "language": doc.language,
            },
            ingested_at=datetime.utcnow().isoformat(),
        )
