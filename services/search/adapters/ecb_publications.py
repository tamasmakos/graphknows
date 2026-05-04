"""ECB Economic Bulletin adapter."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from services.search.base_source import BaseSource
from services.search.parsed_document import ParsedDocument

_BASE = "https://www.ecb.europa.eu"
_LISTING = f"{_BASE}/pub/economic-bulletin/html/index.en.html"


class ECBPublicationsSource(BaseSource):
    source_id = "ecb_publications"
    source_type = "ECB_ECONOMIC_BULLETIN"
    institution = "European Central Bank"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        try:
            html = await self._get(_LISTING)
        except Exception as exc:
            print(f"[ecb_publications] listing failed: {exc}")
            return []

        pattern = re.compile(
            r'href=["\']([^"\']*?/pub/economic-bulletin/html/eb[^"\']*?\.en\.html)["\']',
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
                doc = await self.fetch_one(url)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                print(f"[ecb_publications] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument | None:
        try:
            html = await self._get(url)
        except Exception as exc:
            print(f"[ecb_publications] HTTP error for {url}: {exc}")
            return None

        # Title from page
        title_m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', title_m.group(1))).strip() if title_m else "ECB Economic Bulletin"

        date = self._extract_date(html, url)

        # Extract article sections as H2 headings
        # Look for section headings in ECB bulletin structure
        sections_md = self._extract_sections(html)
        markdown_body = sections_md or self._html_to_markdown(html)
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

    def _extract_sections(self, html: str) -> str:
        """Extract named sections from ECB bulletin HTML."""
        # ECB bulletins use <article> or <section> tags with headings
        sections = []
        # Look for h2/h3 headings and the text after them
        heading_pattern = re.compile(r'<h[23][^>]*>(.*?)</h[23]>', re.IGNORECASE | re.DOTALL)
        para_pattern = re.compile(r'<p[^>]*>(.*?)</p>', re.IGNORECASE | re.DOTALL)

        headings = heading_pattern.findall(html)
        for h in headings:
            clean_h = re.sub(r'<[^>]+>', '', h).strip()
            if clean_h:
                sections.append(f"## {clean_h}")

        if not sections:
            return ""

        # Pair headings with paragraph text (simplified: just return headings + all paragraphs)
        paras = para_pattern.findall(html)
        para_text = "\n\n".join(re.sub(r'<[^>]+>', '', p).strip() for p in paras if re.sub(r'<[^>]+>', '', p).strip())
        return "\n\n".join(sections) + "\n\n" + para_text

    def _extract_date(self, html: str, url: str) -> str:
        # Try page meta/time tags
        for pattern in [
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'<time[^>]+datetime=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'([\d]{4}-[\d]{2}-[\d]{2})',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                return m.group(1)
        # Try URL (e.g. eb202301~...)
        m = re.search(r'eb(\d{4})(\d{2})', url)
        if m:
            return f"{m.group(1)}-{m.group(2)}-01"
        return datetime.utcnow().strftime("%Y-%m-%d")
