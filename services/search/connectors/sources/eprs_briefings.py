"""EPRS Briefings connector — EP Research Service Think Tank."""

from __future__ import annotations

import re
from urllib.parse import urljoin

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

_BASE = "https://www.europarl.europa.eu"
_LISTING = f"{_BASE}/thinktank/en/research/advanced-search?category=briefing&documentType=BRIE"


class EPRSBriefingsConnector(ScrapingConnector):
    """Scrapes EP Research Service briefings from the Think Tank portal."""

    source_id = "eprs_briefings"
    connector_type = ConnectorType.HTML_SCRAPE
    institution = "EP Research Service"
    source_type = "EPRS_BRIEFING"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=1.0, timeout=30)

    async def _scrape(self, limit: int = 10) -> list[ParsedDocument]:
        html = await self._get(_LISTING)

        seen: dict[str, bool] = {}
        pattern = re.compile(
            r'href=["\']([^"\']*?/thinktank/en/document/[^"\']+)["\']', re.IGNORECASE
        )
        for m in pattern.findall(html):
            url = m if m.startswith("http") else urljoin(_BASE, m)
            seen[url] = True
            if len(seen) >= limit:
                break

        if not seen:
            fallback = re.compile(r'href=["\']([^"\']*?/thinktank/[^"\']+)["\']', re.IGNORECASE)
            for m in fallback.findall(html):
                if "search.html" in m or "category" in m:
                    continue
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
                    "[eprs_briefings] fetch_one failed %s: %s", url, exc
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
            else "EPRS Briefing"
        )
        date = self._extract_date(html)
        markdown_body = self._html_to_markdown(html)
        markdown = f"# {title}\n\n{markdown_body}"
        # Strip nav/boilerplate tags before building plain-text version
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(['nav', 'footer', 'header', 'aside', 'script', 'style']):
                tag.decompose()
            clean_html = str(soup)
        except ImportError:
            import re as _re
            clean_html = _re.sub(
                r'<(nav|footer|header|aside|script|style)[^>]*>.*?</\1>',
                '', html, flags=_re.DOTALL | _re.IGNORECASE,
            )
        raw_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", clean_html)).strip()
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
