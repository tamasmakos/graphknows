"""Council Conclusions adapter – scrapes European Council meeting conclusions PDFs."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

_BASE = "https://www.consilium.europa.eu"
_LISTING = f"{_BASE}/en/meetings/european-council/"


class CouncilConclusionsSource(BaseSource):
    source_id = "council_conclusions"
    source_type = "COUNCIL_CONCLUSION"
    institution = "European Council"

    async def scrape(self, limit: int = 5) -> list[ParsedDocument]:
        try:
            html = await self._get(_LISTING)
        except Exception as exc:
            raise NotImplementedError(  # TODO: implement council_conclusions adapter
                f"[council_conclusions] listing failed: {exc}"
            ) from exc

        # Find PDF links that contain /media/ in the href
        pattern = re.compile(r'href=["\']([^"\']*?/media/[^"\']*?\.pdf)["\']', re.IGNORECASE)
        matches = pattern.findall(html)

        seen: dict[str, bool] = {}
        for m in matches:
            url = m if m.startswith("http") else urljoin(_BASE, m)
            seen[url] = True
            if len(seen) >= limit:
                break

        docs: list[ParsedDocument] = []
        for url in list(seen.keys())[:limit]:
            try:
                doc = await self.fetch_one(url)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                print(f"[council_conclusions] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument | None:
        try:
            # _get returns text; we need bytes for PDF - use client directly
            if self._client:
                resp = await self._client.get(url)
                resp.raise_for_status()
                pdf_bytes = resp.content
            else:
                import httpx
                async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as c:
                    resp = await c.get(url)
                    resp.raise_for_status()
                    pdf_bytes = resp.content
        except Exception as exc:
            print(f"[council_conclusions] HTTP error for {url}: {exc}")
            return None

        markdown_body = self._pdf_to_markdown(pdf_bytes)

        # Extract title from URL or use default
        title = self._title_from_url(url)

        # Extract date from URL
        date = self._date_from_url(url) or datetime.utcnow().strftime("%Y-%m-%d")

        markdown = f"# {title}\n\n{markdown_body}"
        raw_text = re.sub(r'\s+', ' ', markdown_body).strip()
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
        # Try to get a readable name from the URL filename
        filename = url.rstrip("/").split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
        return filename.title() or "European Council Conclusions"

    def _date_from_url(self, url: str) -> str | None:
        # Look for YYYY-MM-DD or YYYYMMDD in URL
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = re.search(r'(\d{4})(\d{2})(\d{2})', url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = re.search(r'(\d{4})', url)
        if m:
            return f"{m.group(1)}-01-01"
        return None
