"""Tavily web search tool — general + news search for the EU investigative agent.

Provides a normalised document list identical in shape to the EU adapter output,
so the ranking layer and graphgen pipeline can consume both interchangeably.

EU-specific domains are boosted via ``include_domains`` when the query contains
EU keywords; for purely general queries we use Tavily's news topic to get
recent coverage as well.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from agent.tools.base import BaseTool

log = logging.getLogger("agent.tools.tavily")

# Authoritative EU public domains — weight these results higher downstream
EU_DOMAINS: list[str] = [
    "europarl.europa.eu",
    "ec.europa.eu",
    "consilium.europa.eu",
    "eur-lex.europa.eu",
    "publications.europa.eu",
    "eeas.europa.eu",
    "ecb.europa.eu",
    "eca.europa.eu",
    "ombudsman.europa.eu",
    "edps.europa.eu",
]

_EU_KEYWORDS = {
    "eu",
    "european",
    "parliament",
    "commission",
    "council",
    "regulation",
    "directive",
    "treaty",
    "cellar",
    "celex",
    "eur-lex",
    "eurozone",
    "schengen",
    "ecb",
    "euro",
    "brussels",
}


def _is_eu_query(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & _EU_KEYWORDS)


class TavilySearchTool(BaseTool):
    """Searches the web via Tavily and returns normalised document dicts.

    Args:
        api_key: Tavily API key. Defaults to TAVILY_API_KEY env var.
    """

    name = "tavily_search"
    description = (
        "Search the live web for general context, news, and background on any topic. "
        'Params: {"query": "...", "max_results": 8, "topic": "general|news", '
        '"time_range": "day|week|month|year"}'
    )

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key or os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API", "")
        )

    async def run(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            return {"error": "TAVILY_API_KEY not configured", "documents": [], "count": 0}

        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "query is required", "documents": [], "count": 0}

        max_results = min(int(params.get("max_results", 8) or 8), 20)
        topic = str(params.get("topic", "general"))
        time_range = params.get("time_range")  # None = no filter

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._api_key)

            # Run two parallel searches: general + news for EU topics
            searches: list[Any] = [
                client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",
                    topic=topic,
                    time_range=time_range,
                    include_raw_content=True,
                    include_answer=False,
                )
            ]
            if _is_eu_query(query) and topic != "news":
                # Also search EU authoritative domains specifically
                searches.append(
                    client.search(
                        query=query,
                        max_results=max_results // 2,
                        search_depth="advanced",
                        topic="general",
                        include_domains=EU_DOMAINS,
                        include_raw_content=True,
                        include_answer=False,
                    )
                )

            responses = await asyncio.gather(*searches, return_exceptions=True)

            seen_urls: set[str] = set()
            documents: list[dict[str, Any]] = []

            for resp in responses:
                if isinstance(resp, Exception):
                    log.warning("Tavily search failed: %s", resp)
                    continue
                for r in resp.get("results", []):
                    url = r.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    text = r.get("raw_content") or r.get("content") or ""
                    documents.append(
                        {
                            "id": f"tavily:{url}",
                            "title": r.get("title", url),
                            "text": text[:8000],  # cap per-doc
                            "url": url,
                            "source": "tavily",
                            "date": r.get("published_date", ""),
                            "score": float(r.get("score", 0.0)),
                        }
                    )

            log.info("Tavily: %d documents fetched for %r", len(documents), query)
            return {
                "documents": documents,
                "count": len(documents),
                "total_fetched": len(documents),
            }

        except Exception as exc:
            log.exception("TavilySearchTool.run failed")
            return {"error": str(exc), "documents": [], "count": 0}
