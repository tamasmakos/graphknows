"""Council Conclusions connector — European Council meeting PDFs."""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

_BASE = "https://www.consilium.europa.eu"
_LISTING = f"{_BASE}/en/meetings/european-council/"


class CouncilConclusionsConnector(ScrapingConnector):
    """Scrapes European Council conclusions from PDF links."""

    source_id = "council_conclusions"
    connector_type = ConnectorType.HTML_SCRAPE
    institution = "European Council"
    source_type = "COUNCIL_CONCLUSION"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=1.0, timeout=45)

    async def _scrape(self, limit: int = 5) -> list[ParsedDocument]:
        html = await self._get(_LISTING)

        pattern = re.compile(r'href=["\']([^"\']*?/media/[^"\']*?\.pdf)["\']', re.IGNORECASE)
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
                    "[council_conclusions] fetch_one failed %s: %s", url, exc
                )
        return docs

    async def _fetch_one(self, url: str) -> ParsedDocument | None:
        client = self._ensure_client()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("[council_conclusions] HTTP error %s: %s", url, exc)
            return None

        markdown_body = self._pdf_to_markdown(pdf_bytes)
        title = self._title_from_url(url)
        date = self._date_from_url(url) or datetime.utcnow().strftime("%Y-%m-%d")
        markdown = f"# {title}\n\n{markdown_body}"
        raw_text = re.sub(r"\s+", " ", markdown_body).strip()
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

    def _title_from_url(self, url: str) -> str:
        filename = (
            url.rstrip("/").split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
        )
        return filename.title() or "European Council Conclusions"

    def _date_from_url(self, url: str) -> str | None:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = re.search(r"(\d{4})(\d{2})(\d{2})", url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = re.search(r"/(20\d{2})/", url)
        if m:
            return f"{m.group(1)}-01-01"
        return None
