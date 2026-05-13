"""EP Resolutions adapter – scrapes European Parliament texts-adopted page."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

_BASE = "https://www.europarl.europa.eu"
_LISTING = f"{_BASE}/plenary/en/texts-adopted.html"


class EPResolutionsSource(BaseSource):
    source_id = "ep_resolutions"
    source_type = "EP_RESOLUTION"
    institution = "European Parliament"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        """Fetch the texts-adopted listing and retrieve up to `limit` resolutions."""
        html = await self._get(_LISTING)

        # Find links to individual resolution pages
        # EP links look like: /doceo/document/TA-..._EN.html
        pattern = re.compile(
            r'href=["\']([^"\']*?/doceo/document/TA-[^"\']+?_EN\.html)["\']',
            re.IGNORECASE,
        )
        matches = pattern.findall(html)
        seen: dict[str, str] = {}
        for m in matches:
            url = m if m.startswith("http") else urljoin(_BASE, m)
            if url not in seen:
                seen[url] = url
            if len(seen) >= limit:
                break

        docs: list[ParsedDocument] = []
        for url in list(seen.keys())[:limit]:
            try:
                doc = await self.fetch_one(url)
                docs.append(doc)
            except Exception as exc:
                # log and continue
                print(f"[ep_resolutions] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument:
        """Fetch a single EP resolution page and return a ParsedDocument."""
        html = await self._get(url)

        # --- title ---
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        raw_title = title_match.group(1).strip() if title_match else "EP Resolution"
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', raw_title)).strip()

        # --- date: look for ISO date patterns or EP formatted dates in page ---
        date = self._extract_date(html)

        # --- convert to markdown ---
        markdown_body = self._html_to_markdown(html)

        # Build structured markdown: H1 = title, rest of body follows
        markdown = f"# {title}\n\n{markdown_body}"

        raw_text = re.sub(r'<[^>]+>', ' ', html)
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()

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

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _extract_date(self, html: str) -> str:
        """Try to extract an ISO date from the page."""
        # datetime meta tag
        for pattern in [
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\'](\d{4}-\d{2}-\d{2})',
            r'<time[^>]+datetime=["\'](\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2})',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                return m.group(1)
        return datetime.utcnow().strftime("%Y-%m-%d")
