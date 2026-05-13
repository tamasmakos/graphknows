"""Wikipedia EU topics adapter — fetches Wikipedia articles on EU institutions and policies."""
from __future__ import annotations

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

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

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_FULL = "https://en.wikipedia.org/w/api.php"


class WikipediaEUSource(BaseSource):
    """Adapter that fetches Wikipedia articles for key EU policy topics.

    Attributes:
        source_id: Identifier for this adapter.
        source_type: Document type label.
        institution: Publishing institution label.
    """

    source_id = "wikipedia_eu"
    source_type = "WIKIPEDIA_ARTICLE"
    institution = "Wikipedia"

    async def scrape(self, limit: int = 10) -> list[ParsedDocument]:
        """Fetch up to *limit* Wikipedia articles about EU topics.

        Args:
            limit: Maximum number of articles to retrieve.

        Returns:
            List of successfully parsed documents.
        """
        docs: list[ParsedDocument] = []
        for title in EU_TOPICS[:limit]:
            doc = await self.fetch_one(title)
            if doc:
                docs.append(doc)
        return docs

    async def fetch_one(self, url: str) -> ParsedDocument | None:  # type: ignore[override]
        """Fetch a single Wikipedia article by title or URL.

        Args:
            url: Wikipedia article title (e.g. ``EU_AI_Act``) or full URL.

        Returns:
            A ``ParsedDocument``, or ``None`` if the page is missing/too short.
        """
        title = url.split("/wiki/")[-1] if "/wiki/" in url else url
        params = {
            "action": "query",
            "titles": title.replace("_", " "),
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        try:
            assert self._client is not None, "Client not initialised — use async context manager"
            r = await self._client.get(WIKI_FULL, params=params)
            r.raise_for_status()
            data = r.json()
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page:
                return None
            text: str = page.get("extract", "")
            if len(text) < 200:
                return None

            # Summary for title + timestamp
            r2 = await self._client.get(WIKI_API.format(title=title))
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
            import logging
            logging.getLogger(__name__).warning("[wikipedia_eu] failed for %s: %s", title, exc)
            return None
