"""EP Resolutions connector — live scraper for plenary texts adopted."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

_BASE = "https://www.europarl.europa.eu"
_LISTING = f"{_BASE}/plenary/en/texts-adopted.html"


class EPResolutionsConnector(ScrapingConnector):
    """Scrapes EP adopted texts (non-legislative resolutions)."""

    source_id = "ep_resolutions"
    connector_type = ConnectorType.HTML_SCRAPE
    institution = "European Parliament"
    source_type = "EP_RESOLUTION"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=1.0, timeout=30)

    async def _scrape(self, limit: int = 10) -> list[ParsedDocument]:
        html = await self._get(_LISTING)

        pattern = re.compile(
            r'href=["\']([^"\']*?/doceo/document/TA[^"\']*?_EN\.html[^"\']*?)["\']',
            re.IGNORECASE,
        )
        seen: dict[str, bool] = {}
        for m in pattern.findall(html):
            url = m if m.startswith("http") else urljoin(_BASE, m)
            seen[url] = True
            if len(seen) >= limit:
                break

        docs: list[ParsedDocument] = []
        for url in list(seen.keys())[:limit]:
            try:
                doc = await self._fetch_one(url)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "[ep_resolutions] fetch_one failed for %s: %s", url, exc
                )
        return docs

    async def _fetch_one(self, url: str) -> ParsedDocument | None:
        try:
            html = await self._get(url)
        except Exception:
            return None

        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = (
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
            if title_m
            else "EP Resolution"
        )
        date = self._extract_date(html, url)
        markdown = self._html_to_markdown(html)
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
