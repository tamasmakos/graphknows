"""Wikipedia EU connector — key EU policy topics from Wikipedia."""
from __future__ import annotations

import logging

from ..base import ConnectorType, ScrapingConnector
from ...parsed_document import ParsedDocument

log = logging.getLogger(__name__)

EU_TOPICS = [
    "European_Union_sanctions_against_Russia",
    "EU_AI_Act",
    "European_Green_Deal",
    "Migration_policy_of_the_European_Union",
    "Common_Agricultural_Policy",
    "Enlargement_of_the_European_Union",
    "European_Defence_Agency",
    "Rule_of_law_in_the_European_Union",
    "Digital_Services_Act",
    "Digital_Markets_Act",
]

_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_FULL = "https://en.wikipedia.org/w/api.php"


class WikipediaEUConnector(ScrapingConnector):
    """Fetches Wikipedia articles for key EU policy topics."""

    source_id = "wikipedia_eu"
    connector_type = ConnectorType.REST_API
    institution = "Wikipedia"
    source_type = "WIKIPEDIA_ARTICLE"

    def __init__(self) -> None:
        super().__init__(rate_limit_rps=2.0, timeout=20)

    async def _scrape(self, limit: int = 10) -> list[ParsedDocument]:
        docs: list[ParsedDocument] = []
        client = self._ensure_client()
        for topic in EU_TOPICS[:limit]:
            doc = await self._fetch_wiki(client, topic)
            if doc:
                docs.append(doc)
        return docs

    async def _fetch_one(self, title: str) -> ParsedDocument | None:
        """title may be a Wikipedia article title or page slug."""
        if "/wiki/" in title:
            title = title.split("/wiki/")[-1]
        client = self._ensure_client()
        return await self._fetch_wiki(client, title)

    async def _fetch_wiki(self, client, title: str) -> ParsedDocument | None:
        import httpx
        params = {
            "action": "query",
            "titles": title.replace("_", " "),
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        try:
            r = await client.get(_WIKI_FULL, params=params)
            r.raise_for_status()
            data = r.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                return None
            text: str = page.get("extract", "")
            if len(text) < 200:
                return None

            r2 = await client.get(_WIKI_SUMMARY.format(title=title))
            summary = r2.json() if r2.status_code == 200 else {}

            doc_id = self._make_doc_id("wikipedia", title)
            article_title: str = summary.get("title", title.replace("_", " "))
            date: str = (summary.get("timestamp", "") or "")[:10] or "2024-01-01"

            return ParsedDocument(
                source_id=self.source_id,
                source_type=self.source_type,
                url=f"https://en.wikipedia.org/wiki/{title}",
                title=article_title,
                date=date,
                institution=self.institution,
                language="en",
                raw_text=text[:50_000],
                markdown=f"# {article_title}\n\n{text[:50_000]}",
                doc_id=doc_id,
                metadata={"wikipedia_title": title},
            )
        except Exception as exc:
            log.warning("[wikipedia_eu] failed for %s: %s", title, exc)
            return None
