"""Cellar SPARQL connector — EU Publications Office CDM endpoint.

Upgraded from the original adapter to support dynamic date_from/date_to
parameters. The SPARQL endpoint URL is hardcoded (SSRF prevention).
"""

from __future__ import annotations

import logging

from ..base import ConnectorType, ScrapingConnector
from ...models import SearchQuery
from ...parsed_document import ParsedDocument

log = logging.getLogger(__name__)

# Endpoint is allowlisted — user input never becomes a URL (SSRF prevention)
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

_DOC_TYPE_MAP: dict[str, str] = {
    "regulation": "cdm:regulation",
    "press_release": "cdm:press_release",
    "decision": "cdm:decision",
    "conclusions_council": "cdm:conclusions_council",
    "work": "cdm:work",
}


class CellarSPARQLConnector(ScrapingConnector):
    """Queries the EU CELLAR SPARQL endpoint for legislation and press releases."""

    source_id = "cellar_sparql"
    connector_type = ConnectorType.SPARQL
    institution = "EU Publications Office"

    def __init__(self, doc_type: str = "press_release") -> None:
        super().__init__(rate_limit_rps=0.5, timeout=30)
        self._doc_type = doc_type
        self.source_type = doc_type.upper()

    async def _scrape(self, limit: int = 5) -> list[ParsedDocument]:
        """Run a generic SPARQL scrape (no date filter — used by search())."""
        return await self._run_sparql(limit=limit)

    async def search(self, query: SearchQuery) -> list[SearchResult]:  # type: ignore[override]
        """Override to pass date_from/date_to into SPARQL filter."""
        from ...models import SearchResult

        docs = await self._run_sparql(
            limit=query.limit * 2,
            date_from=query.date_from,
            date_to=query.date_to,
        )
        q_lower = query.query.lower()
        results: list[SearchResult] = []
        for doc in docs:
            combined = (doc.title or "").lower() + " " + (doc.raw_text or "").lower()
            if q_lower and q_lower not in combined:
                continue
            self._doc_cache[doc.doc_id] = doc
            results.append(
                SearchResult(
                    id=doc.doc_id,
                    source=self.source_id,
                    title=doc.title,
                    date=doc.date or None,
                    url=doc.url,
                    excerpt=(doc.raw_text or "")[:300] or None,
                )
            )
            if len(results) >= query.limit:
                break
        return results

    async def _run_sparql(
        self,
        limit: int = 5,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[ParsedDocument]:
        cdm_class = _DOC_TYPE_MAP.get(self._doc_type, "cdm:press_release")
        date_filter = _build_date_filter(date_from, date_to)

        sparql = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        SELECT ?work ?title ?date WHERE {{
          ?work a {cdm_class} .
          ?work cdm:work_date_document ?date .
          OPTIONAL {{
            ?work cdm:work_has_expression ?expr .
            ?expr cdm:expression_uses_language
                  <http://publications.europa.eu/resource/authority/language/ENG> .
            ?expr cdm:expression_title ?title .
          }}
          {date_filter}
          FILTER(BOUND(?title))
        }} ORDER BY DESC(?date) LIMIT {limit}
        """
        client = self._ensure_client()
        try:
            r = await client.post(
                SPARQL_ENDPOINT,
                data={"query": sparql, "format": "application/sparql-results+json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            r.raise_for_status()
            bindings: list[dict] = r.json()["results"]["bindings"]
        except Exception as exc:
            log.warning("[cellar_sparql] SPARQL query failed: %s", exc)
            return []

        docs: list[ParsedDocument] = []
        for b in bindings:
            title: str = b.get("title", {}).get("value", "Untitled")
            date: str = b.get("date", {}).get("value", "")[:10]
            work_uri: str = b.get("work", {}).get("value", "")
            uuid: str = work_uri.split("/")[-1] if "/" in work_uri else ""
            url = (
                f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=cellar:{uuid}"
                if uuid
                else work_uri
            )
            doc_id = self._make_doc_id("cellar", work_uri)
            docs.append(
                ParsedDocument(
                    source_id=self.source_id,
                    source_type=self.source_type,
                    url=url,
                    title=title,
                    date=date,
                    institution=self.institution,
                    language="en",
                    raw_text=title,
                    markdown=f"# {title}\n\nDate: {date}\nSource: {url}\n",
                    doc_id=doc_id,
                    metadata={"cellar_uri": work_uri, "date": date},
                )
            )
        return docs

    async def _fetch_one(self, url: str) -> ParsedDocument | None:
        """Not implemented — EUR-Lex full text requires authentication."""
        return None


def _build_date_filter(date_from: str | None, date_to: str | None) -> str:
    xsd = "<http://www.w3.org/2001/XMLSchema#date>"
    parts: list[str] = []
    if date_from:
        parts.append(f'FILTER(?date >= "{date_from}"^^{xsd})')
    if date_to:
        parts.append(f'FILTER(?date <= "{date_to}"^^{xsd})')
    if not parts:
        # Default: last 2 years
        from datetime import date, timedelta

        cutoff = (date.today() - timedelta(days=730)).isoformat()
        parts.append(f'FILTER(?date >= "{cutoff}"^^{xsd})')
    return "\n          ".join(parts)
