"""Abstract BaseSource for Graphknows ingestion pipeline."""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from services.search.parsed_document import ParsedDocument

_USER_AGENT = 'GraphknowsBot/1.0 (political-intelligence-research)'


class BaseSource(ABC):
    source_id: str
    source_type: str
    institution: str

    def __init__(self, rate_limit_rps: float = 1.0, timeout: int = 30):
        self._rate_limit_rps = rate_limit_rps
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(1)
        self._min_interval = 1.0 / rate_limit_rps
        self._last_request: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "BaseSource":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def scrape(self, limit: int = 10) -> list[ParsedDocument]: ...

    @abstractmethod
    async def fetch_one(self, url: str) -> ParsedDocument: ...

    # ------------------------------------------------------------------ #
    # HTTP helper
    # ------------------------------------------------------------------ #

    async def _get(self, url: str, **kwargs) -> str:
        """Rate-limited GET with 3 retries (1s, 2s, 4s exponential backoff)."""
        client = self._client
        if client is None:
            client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            )

        async with self._semaphore:
            # enforce rate limit
            elapsed = time.monotonic() - self._last_request
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

        delays = [1, 2, 4]
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt, delay in enumerate(delays):
            try:
                resp = await client.get(url, **kwargs)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                last_exc = exc
                if attempt < len(delays) - 1:
                    await asyncio.sleep(delay)
        raise last_exc

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    def _html_to_markdown(self, html: str) -> str:
        try:
            import markdownify
            md = markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            # basic tag stripping fallback
            md = re.sub(r'<[^>]+>', ' ', html)
        # clean up excessive blank lines
        md = re.sub(r'\n{3,}', '\n\n', md).strip()
        return md

    def _pdf_to_markdown(self, pdf_bytes: bytes) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"## Page {i}\n\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            return ""

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text[:2000])
        except Exception:
            return "en"

    def _make_doc_id(self, *parts: str) -> str:
        return hashlib.sha1(":".join(parts).encode()).hexdigest()[:16]
