"""CELLAR SPARQL tool — fetches recent EU legislation metadata for the investigative agent.

Queries the EU Publications Office SPARQL endpoint to retrieve CELEX IDs and dates
for recent EU Regulations/Directives/Decisions (2018–present), then returns normalised
document dicts matching the TavilySearchTool output shape.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
import urllib.request
import json
from typing import Any

from agent.tools.base import BaseTool

log = logging.getLogger("agent.tools.cellar")

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
_HEADERS = {
    "User-Agent": "Graphknows/1.0 (https://graphknows.io; research@graphknows.io)",
    "Accept": "application/sparql-results+json",
}
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "from", "has", "have",
    "been", "its", "not", "but", "was", "you", "they", "their", "will", "more",
    "about", "also", "can", "all", "any", "into", "than", "such", "when",
}


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query string.

    Args:
        query: Raw query string.

    Returns:
        List of lowercase keywords with 3+ chars, excluding stopwords.
    """
    tokens = query.lower().split()
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _sparql_query_sync(sparql: str, timeout: int = 10) -> list[dict]:
    """Execute a SPARQL query synchronously via urllib.

    Args:
        sparql: SPARQL query string.
        timeout: Request timeout in seconds.

    Returns:
        List of binding dicts from the SPARQL results.
    """
    data = urllib.parse.urlencode({"query": sparql}).encode()
    req = urllib.request.Request(SPARQL_ENDPOINT, data=data, headers=_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
    return result.get("results", {}).get("bindings", [])


class CellarSPARQLTool(BaseTool):
    """Queries CELLAR SPARQL endpoint for recent EU legislation metadata.

    Returns normalised document dicts with the same shape as TavilySearchTool output
    so they can be consumed interchangeably by the investigative pipeline.
    """

    name = "cellar_sparql"
    description = (
        "Query EU CELLAR SPARQL endpoint for recent EU Regulations/Directives/Decisions. "
        'Params: {"query": "...", "limit": 20}'
    )

    async def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch recent EU legislation metadata from CELLAR SPARQL.

        Args:
            params: Dict with optional 'query' and 'limit' keys.

        Returns:
            Dict with 'documents' list, 'count', and optional 'error'.
        """
        query = str(params.get("query", "")).strip()
        limit = int(params.get("limit", 20))
        keywords = _extract_keywords(query) if query else []

        sparql = f"""PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex ?date WHERE {{
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_date_document ?date .
  FILTER(REGEX(STR(?celex), '^3202[0-9]R[0-9]{{4}}$'))
}} ORDER BY DESC(?date) LIMIT {limit}"""

        try:
            bindings = await asyncio.to_thread(_sparql_query_sync, sparql, 10)
        except Exception as exc:
            log.warning("CellarSPARQLTool: SPARQL query failed: %s", exc)
            return {"documents": [], "count": 0, "error": str(exc)}

        documents: list[dict[str, Any]] = []
        for b in bindings:
            celex = b.get("celex", {}).get("value", "")
            date = b.get("date", {}).get("value", "")[:10]
            if not celex:
                continue

            eur_lex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
            cellar_url = f"https://publications.europa.eu/resource/celex/{celex}"
            text = f"EU Regulation {celex} adopted {date}. Primary source: {eur_lex_url}"

            # Keyword relevance filter: if keywords provided, skip docs that don't match any
            # (CELLAR can't do full-text so we do best-effort CELEX-level filtering)
            if keywords:
                celex_lower = celex.lower()
                if not any(kw in celex_lower or kw in date for kw in keywords):
                    # Still include — CELEX IDs rarely match keywords; include all and let
                    # downstream ranking sort by relevance
                    pass

            documents.append({
                "id": f"cellar:{celex}",
                "title": f"EU Regulation {celex}",
                "text": text,
                "url": eur_lex_url,
                "source": "cellar",
                "date": date,
                "score": 0.5,
                "metadata": {
                    "source": "cellar",
                    "celex": celex,
                    "cellar_url": cellar_url,
                },
            })

        log.info("CellarSPARQLTool: %d documents fetched for %r", len(documents), query)
        return {"documents": documents, "count": len(documents)}
