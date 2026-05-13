"""CELLAR SPARQL adapter — queries EU Publications Office for regulations, press releases, and more."""
from __future__ import annotations

import logging

from ..base_source import BaseSource
from ..parsed_document import ParsedDocument

log = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

_DOC_TYPE_MAP: dict[str, str] = {
    "regulation": "cdm:regulation",
    "press_release": "cdm:press_release",
    "decision": "cdm:decision",
    "conclusions_council": "cdm:conclusions_council",
    "work": "cdm:work",
}


class CellarSPARQLSource(BaseSource):
    """Adapter that runs SPARQL queries against the EU CELLAR endpoint.

    Attributes:
        source_id: Identifier for this adapter.
        source_type: Document type label (overridden per *doc_type*).
        institution: Publishing institution label.
    """

    source_id = "cellar_sparql"
    source_type = "EU_DOCUMENT"
    institution = "EU Publications Office"

    def __init__(self, doc_type: str = "press_release", **kwargs) -> None:
        """Initialise the adapter.

        Args:
            doc_type: One of ``regulation``, ``press_release``, ``decision``,
                ``conclusions_council``.
            **kwargs: Forwarded to :class:`BaseSource`.
        """
        super().__init__(**kwargs)
        self._doc_type = doc_type
        self.source_type = doc_type.upper()

    async def scrape(self, limit: int = 5) -> list[ParsedDocument]:
        """Fetch metadata records from CELLAR via SPARQL.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of :class:`ParsedDocument` objects (metadata-only).
        """
        cdm_class = _DOC_TYPE_MAP.get(self._doc_type, "cdm:press_release")
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
          FILTER(?date > "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#date>)
          FILTER(BOUND(?title))
        }} ORDER BY DESC(?date) LIMIT {limit}
        """
        try:
            assert self._client is not None, "Client not initialised — use async context manager"
            r = await self._client.post(
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

    async def fetch_one(self, url: str) -> ParsedDocument | None:
        """Fetch full text for a single document (not implemented — EUR-Lex blocks IPs).

        Args:
            url: Document URL.

        Returns:
            Always ``None``; full-text fetch is not supported.
        """
        return None
