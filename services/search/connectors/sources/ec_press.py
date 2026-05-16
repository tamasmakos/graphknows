"""EC Press connector — European Commission press corner."""
from __future__ import annotations

import json
import re
from datetime import datetime
from urllib.parse import urljoin

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

_BASE = "https://ec.europa.eu"
_API_URL = (
    f"{_BASE}/commission/presscorner/api/items"
    "?keywords=&dateTo=&dateFrom=&lang=EN&limit=20&topic="
)
_HTML_LISTING = f"{_BASE}/commission/presscorner/home/en"


class ECPressConnector(ScrapingConnector):
    """Scrapes European Commission press releases. Prefers JSON API."""

    source_id = "ec_press"
    connector_type = ConnectorType.REST_API
    institution = "European Commission"
    source_type = "EC_PRESS_RELEASE"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=1.0, timeout=30)

    async def _scrape(self, limit: int = 10) -> list[ParsedDocument]:
        try:
            text = await self._get(_API_URL)
            data = json.loads(text)
            items = (
                data.get("items")
                or data.get("results")
                or (data if isinstance(data, list) else [])
            )
            docs: list[ParsedDocument] = []
            for item in items[:limit]:
                url = item.get("url") or item.get("link") or ""
                if not url:
                    continue
                if not url.startswith("http"):
                    url = urljoin(_BASE, url)
                doc = await self._fetch_one(url, prefill=item)
                if doc:
                    docs.append(doc)
            return docs
        except Exception:
            return await self._scrape_html(limit)

    async def _scrape_html(self, limit: int) -> list[ParsedDocument]:
        try:
            html = await self._get(_HTML_LISTING)
        except Exception:
            return []

        pattern = re.compile(
            r'href=["\']([^"\']*?/presscorner/detail/[^"\']+)["\']', re.IGNORECASE
        )
        seen: dict[str, bool] = {}
        for m in pattern.findall(html):
            url = m if m.startswith("http") else urljoin(_BASE, m)
            seen[url] = True
            if len(seen) >= limit:
                break

        docs: list[ParsedDocument] = []
        for url in list(seen.keys())[:limit]:
            doc = await self._fetch_one(url)
            if doc:
                docs.append(doc)
        return docs

    async def _fetch_one(self, url: str, prefill: dict | None = None) -> ParsedDocument | None:
        prefill = prefill or {}
        try:
            html = await self._get(url)
        except Exception:
            return None

        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = (
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
            if title_m
            else prefill.get("title", "EC Press Release")
        )
        date = (
            prefill.get("date")
            or self._extract_date(html)
            or datetime.utcnow().strftime("%Y-%m-%d")
        )
        if isinstance(date, str) and len(date) > 10:
            date = date[:10]

        markdown_body = self._html_to_markdown(html)
        markdown = f"# {title}\n\n{markdown_body}"
        raw_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        language = self._detect_language(raw_text[:2000])
        doc_id = self._make_doc_id(self.source_id, url)

        return ParsedDocument(
            source_id=self.source_id,
            source_type=self.source_type,
            url=url,
            title=title,
            date=date,
            institution=self.institution,
            language=language,
            raw_text=raw_text[:50000],
            markdown=markdown,
            doc_id=doc_id,
            metadata={"fetched_from": url},
        )
