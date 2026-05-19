"""CELLAR SPARQL tool — 2-stage pipeline: SPARQL title search + full-text fetch.

Stage 1: Query EU Publications Office SPARQL endpoint for CELEX IDs + Cellar URIs.
Stage 2: Content-negotiate XHTML from each Cellar URI and strip to plain text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

from agent.tools.base import BaseTool

log = logging.getLogger("agent.tools.cellar")

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
_SPARQL_HEADERS = {
    "User-Agent": "Graphknows/1.0 (https://graphknows.io; research@graphknows.io)",
    "Accept": "application/sparql-results+json",
    "Content-Type": "application/x-www-form-urlencoded",
}
_FETCH_HEADERS = {
    "User-Agent": "Graphknows/1.0 (https://graphknows.io; research@graphknows.io)",
    "Accept": "application/xhtml+xml",
    "Accept-Language": "en",
}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "are",
    "from",
    "has",
    "have",
    "been",
    "its",
    "not",
    "but",
    "was",
    "you",
    "they",
    "their",
    "will",
    "more",
    "about",
    "also",
    "can",
    "all",
    "any",
    "into",
    "than",
    "such",
    "when",
}


def _extract_keywords(query: str, n: int = 3) -> list[str]:
    tokens = [t.lower() for t in query.split() if len(t) >= 4 and t.lower() not in _STOPWORDS]
    return tokens[:n]


def _sparql_query_sync(sparql: str, timeout: int = 15) -> list[dict]:
    data = urllib.parse.urlencode({"query": sparql}).encode()
    req = urllib.request.Request(SPARQL_ENDPOINT, data=data, headers=_SPARQL_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("results", {}).get("bindings", [])


def _fetch_text_sync(cellar_uri: str, timeout: int = 15) -> str:
    req = urllib.request.Request(cellar_uri, headers=_FETCH_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000]


class CellarSPARQLTool(BaseTool):
    """2-stage CELLAR pipeline: SPARQL title search then full-text content negotiation."""

    name = "cellar_sparql"
    description = "Search EU CELLAR for legislation by keyword and return full-text documents."

    async def run(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch EU legislation with real full text from CELLAR.

        Args:
            params: Dict with 'query' (str) and optional 'limit' (int, default 3, max 5).

        Returns:
            Dict with 'documents' list and 'count'.
        """
        query = str(params.get("query", "")).strip()
        limit = min(int(params.get("limit", 3)), 5)
        keywords = _extract_keywords(query) if query else []

        if not keywords:
            return {"documents": [], "count": 0, "error": "No usable keywords extracted from query"}

        # Build OR filter for all keywords
        kw_filters = " || ".join(f"CONTAINS(LCASE(?title), '{kw}')" for kw in keywords)

        sparql = f"""PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?work ?celex ?title ?date WHERE {{
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_date_document ?date .
  ?expr cdm:expression_belongs_to_work ?work .
  ?expr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> .
  ?expr cdm:expression_title ?title .
  FILTER(REGEX(STR(?celex), '^3202[0-9][RLD][0-9]{{4}}$'))
  FILTER({kw_filters})
}} ORDER BY DESC(?date) LIMIT {limit}"""

        try:
            bindings = await asyncio.to_thread(_sparql_query_sync, sparql, 15)
        except Exception as exc:
            log.warning("CellarSPARQLTool: SPARQL query failed: %s", exc)
            return {"documents": [], "count": 0, "error": str(exc)}

        documents: list[dict[str, Any]] = []
        for b in bindings:
            celex = b.get("celex", {}).get("value", "")
            title = b.get("title", {}).get("value", f"EU Document {celex}")
            date = b.get("date", {}).get("value", "")[:10]
            cellar_uri = b.get("work", {}).get("value", "")
            if not celex or not cellar_uri:
                continue

            # Stage 2: fetch full text
            try:
                full_text = await asyncio.to_thread(_fetch_text_sync, cellar_uri, 15)
            except Exception as exc:
                log.warning("CellarSPARQLTool: text fetch failed for %s: %s", celex, exc)
                full_text = title  # fallback

            eur_lex_url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
            documents.append(
                {
                    "id": f"cellar:{celex}",
                    "title": title,
                    "text": full_text,
                    "url": eur_lex_url,
                    "source": "cellar",
                    "date": date,
                    "score": 0.8,
                }
            )

        log.info("CellarSPARQLTool: %d documents fetched for %r", len(documents), query)
        return {"documents": documents, "count": len(documents)}
