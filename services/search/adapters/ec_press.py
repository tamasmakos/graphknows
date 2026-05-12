"""EC Press Releases adapter – scrapes European Commission press corner."""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

_BASE = "https://ec.europa.eu"
_LISTING = f"{_BASE}/commission/presscorner/api/items?keywords=&dateTo=&dateFrom=&lang=EN&limit=20&topic="


class ECPressSource(BaseSource):
    source_id = "ec_press"
    source_type = "EC_PRESS_RELEASE"
    institution = "European Commission"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        try:
            text = await self._get(_LISTING)
        except Exception as exc:
            print(f"[ec_press] listing failed: {exc}; falling back to HTML scrape")
            return await self._scrape_html(limit)

        try:
            import json
            data = json.loads(text)
            # API may wrap items in various structures
            items = (
                data.get("items")
                or data.get("results")
                or (data if isinstance(data, list) else [])
            )
        except Exception:
            return await self._scrape_html(limit)

        docs: list[ParsedDocument] = []
        for item in items[:limit]:
            url = item.get("url") or item.get("link") or ""
            if not url:
                continue
            if not url.startswith("http"):
                url = urljoin(_BASE, url)
            try:
                doc = await self.fetch_one(url, prefill=item)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                print(f"[ec_press] fetch_one failed for {url}: {exc}")
        return docs

    async def _scrape_html(self, limit: int) -> list[ParsedDocument]:
        """Fallback: scrape the HTML listing page."""
        html_url = f"{_BASE}/commission/presscorner/home/en"
        try:
            html = await self._get(html_url)
        except Exception as exc:
            print(f"[ec_press] HTML fallback also failed: {exc}")
            return []

        pattern = re.compile(r'href=["\']([^"\']*?/presscorner/detail/[^"\']+)["\']', re.IGNORECASE)
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
                print(f"[ec_press] fetch_one failed for {url}: {exc}")
        return docs

    async def fetch_one(self, url: str, prefill: dict | None = None) -> ParsedDocument | None:
        prefill = prefill or {}
        try:
            html = await self._get(url)
        except Exception as exc:
            print(f"[ec_press] HTTP error for {url}: {exc}")
            return None

        # Title
        title_m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', title_m.group(1))).strip() if title_m else prefill.get("title", "EC Press Release")

        # Date
        date = prefill.get("date") or self._extract_date(html) or datetime.utcnow().strftime("%Y-%m-%d")
        if len(date) > 10:
            date = date[:10]

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
            metadata={"fetched_from": url, **{k: v for k, v in prefill.items() if k != "url"}},
        )

    def _extract_date(self, html: str) -> str | None:
        for pattern in [
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'<time[^>]+datetime=["\']([\d]{4}-[\d]{2}-[\d]{2})',
            r'([\d]{4}-[\d]{2}-[\d]{2})',
        ]:
            m = re.search(pattern, html, re.IGNORECASE)
            if m:
                return m.group(1)
        return None
