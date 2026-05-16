"""ECA Reports connector — European Court of Auditors special reports."""
from __future__ import annotations

import re
from urllib.parse import urljoin

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

_BASE = "https://www.eca.europa.eu"
_LISTING = f"{_BASE}/en/publications?category=special_report"


class ECAReportsConnector(ScrapingConnector):
    """Scrapes ECA special reports."""

    source_id = "eca_reports"
    connector_type = ConnectorType.HTML_SCRAPE
    institution = "European Court of Auditors"
    source_type = "ECA_SPECIAL_REPORT"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=1.0, timeout=30)

    async def _scrape(self, limit: int = 10) -> list[ParsedDocument]:
        html = await self._get(_LISTING)

        seen: dict[str, bool] = {}
        for pat in (
            re.compile(r'href=["\']([^"\']*?/en/publications/SR[^"\']*?)["\']', re.IGNORECASE),
            re.compile(r'href=["\']([^"\']*?/en/publications/[^"\']+)["\']', re.IGNORECASE),
        ):
            for m in pat.findall(html):
                if "?category" in m or m.endswith("/publications"):
                    continue
                url = m if m.startswith("http") else urljoin(_BASE, m)
                seen[url] = True
                if len(seen) >= limit:
                    break
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
                    "[eca_reports] fetch_one failed %s: %s", url, exc
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
            else "ECA Special Report"
        )
        date = self._extract_date(html, url)
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
