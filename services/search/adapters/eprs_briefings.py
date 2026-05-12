"""EPRS Briefings adapter – EP Research Service Think Tank."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

_BASE = "https://www.europarl.europa.eu"
_LISTING = f"{_BASE}/thinktank/en/search.html?category=briefing&documentType=BRIE"


class EPRSBriefingsSource(BaseSource):
    source_id = "eprs_briefings"
    source_type = "EPRS_BRIEFING"
    institution = "EP Research Service"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        try:
            html = await self._get(_LISTING)
        except Exception as exc:
            raise NotImplementedError(  # TODO: implement eprs_briefings adapter
                f"[eprs_briefings] listing failed: {exc}"
            ) from exc

        # Extract briefing links from think tank search results
        # URLs like /thinktank/en/document/EPRS_BRI(YYYY)NNNNNN
        pattern = re.compile(
            r'href=["\']([^"\']*?/thinktank/en/document/[^"\']+)["\']',
            re.IGNORECASE,
        )
        seen: dict[str, bool] = {}
        for m in pattern.findall(html):
            url = m if m.startswith("http") else urljoin(_BASE, m)
            seen[url] = True
            if len(seen) >= limit:
                break

        # Fallback: try generic search result links
        if not seen:
            pattern2 = re.compile(r'href=["\']([^"\']*?/thinktank/[^"\']+)["\']', re.IGNORECASE)
            for m in pattern2.findall(html):
                if "search.html" in m or "category" in m:
                    continue
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
                print(f"[eprs_briefings] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument | None:
        try:
            html = await self._get(url)
        except Exception as exc:
            print(f"[eprs_briefings] HTTP error for {url}: {exc}")
            return None

        # Title
        title_m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', title_m.group(1))).strip() if title_m else "EPRS Briefing"

        date = self._extract_date(html)

        # Extract metadata: author, topics
        author = self._extract_author(html)
        topics = self._extract_topics(html)

        markdown_body = self._html_to_markdown(html)
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
            metadata={
                "fetched_from": url,
                "author": author,
                "topics": topics,
            },
        )

    def _extract_author(self, html: str) -> str:
        # Look for author metadata in common patterns
        for pattern in [
            r'<meta[^>]+name=["\']author["\'][^>]+content=["\'](.*?)["\']',
            r'class=["\'][^"\']*author[^"\']*["\'][^>]*>(.*?)</',
        ]:
            m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if m:
                return re.sub(r'<[^>]+>', '', m.group(1)).strip()
        return ""

    def _extract_topics(self, html: str) -> list[str]:
        # Look for topic/tag metadata
        topics = []
        pattern = re.compile(
            r'class=["\'][^"\']*(?:topic|tag|keyword)[^"\']*["\'][^>]*>(.*?)</',
            re.IGNORECASE | re.DOTALL,
        )
        for m in pattern.findall(html):
            t = re.sub(r'<[^>]+>', '', m).strip()
            if t:
                topics.append(t)
        return topics[:10]

    def _extract_date(self, html: str) -> str:
        for pattern in [
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'<time[^>]+datetime=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'([\d]{4}-[\d]{2}-[\d]{2})',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                return m.group(1)
        return datetime.utcnow().strftime("%Y-%m-%d")
