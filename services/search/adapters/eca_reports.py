"""ECA Special Reports adapter – European Court of Auditors."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from services.search.base_source import BaseSource
from services.search.parsed_document import ParsedDocument

_BASE = "https://www.eca.europa.eu"
_LISTING = f"{_BASE}/en/publications?category=special_report"


class ECAReportsSource(BaseSource):
    source_id = "eca_reports"
    source_type = "ECA_SPECIAL_REPORT"
    institution = "European Court of Auditors"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        try:
            html = await self._get(_LISTING)
        except Exception as exc:
            raise NotImplementedError(  # TODO: implement eca_reports adapter
                f"[eca_reports] listing failed: {exc}"
            ) from exc

        # Extract report links – ECA uses /en/publications/SR-YYYY-NN style URLs
        pattern = re.compile(
            r'href=["\']([^"\']*?/en/publications/SR[^"\']*?)["\']',
            re.IGNORECASE,
        )
        # Also try generic publication detail links
        pattern2 = re.compile(
            r'href=["\']([^"\']*?/en/publications/[^"\']+)["\']',
            re.IGNORECASE,
        )

        seen: dict[str, bool] = {}
        for p in (pattern, pattern2):
            for m in p.findall(html):
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
                doc = await self.fetch_one(url)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                print(f"[eca_reports] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument | None:
        try:
            html = await self._get(url)
        except Exception as exc:
            print(f"[eca_reports] HTTP error for {url}: {exc}")
            return None

        # Title
        title_m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', title_m.group(1))).strip() if title_m else "ECA Special Report"

        date = self._extract_date(html, url)

        # Extract summary/findings sections
        markdown_body = self._extract_structured_content(html)
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

    def _extract_structured_content(self, html: str) -> str:
        """Extract summary, key findings, and recommendations."""
        sections = []

        # Look for key ECA report sections
        for section_name in ["Executive summary", "Key findings", "Recommendations", "Introduction", "Observations"]:
            pattern = re.compile(
                rf'({re.escape(section_name)}.*?)</(?:section|div|article)',
                re.IGNORECASE | re.DOTALL,
            )
            m = pattern.search(html)
            if m:
                text = re.sub(r'<[^>]+>', ' ', m.group(1))
                text = re.sub(r'\s+', ' ', text).strip()
                sections.append(f"## {section_name}\n\n{text[:2000]}")

        if sections:
            return "\n\n".join(sections)

        # Fallback: generic HTML to markdown
        return self._html_to_markdown(html)

    def _extract_date(self, html: str, url: str) -> str:
        for pattern in [
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'<time[^>]+datetime=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'([\d]{4}-[\d]{2}-[\d]{2})',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                return m.group(1)
        m = re.search(r'(\d{4})', url)
        if m:
            return f"{m.group(1)}-01-01"
        return datetime.utcnow().strftime("%Y-%m-%d")
