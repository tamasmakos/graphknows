"""Search tool — queries EU public archives + Tavily web search, merges and ranks results."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent.tools.base import BaseTool
from agent.tools.source_router import route_sources
from agent.session.manager import SessionGraphManager

_SERVICES_DIR = Path(__file__).resolve().parents[2]
if str(_SERVICES_DIR) not in sys.path:
    sys.path.append(str(_SERVICES_DIR))

from search.adapters.base import SourceAdapter
from search.adapters.ep_debates import EPDebatesAdapter
from search.adapters.eurlex import EurLexAdapter
from search.adapters.ep_resolutions import EPResolutionsSource
from search.adapters.council_conclusions import CouncilConclusionsSource
from search.adapters.ec_press import ECPressSource
from search.adapters.ecb_publications import ECBPublicationsSource
from search.adapters.eca_reports import ECAReportsSource
from search.adapters.eprs_briefings import EPRSBriefingsSource
from search.adapters.cellar_sparql import CellarSPARQLSource
from search.models import SearchQuery

from agent.tools.ranking import rank_documents


log = logging.getLogger("agent.tools.search")


class SearchTool(BaseTool):
    """Search EU public sources + Tavily web in parallel, merge and rank results.

    EU adapters provide authoritative official records (EP debates, EUR-Lex legislation,
    EC press releases, ECB publications, ECA reports).  Tavily fills the general web
    context gap and surfaces recent news the specialized archives may not yet carry.
    Both result sets are normalised to the same document shape before BM25 ranking.
    """

    name = "search"
    description = (
        "Search EU public archives AND the live web simultaneously. "
        'Params: {"query": "...", "sources": ["ep_debates", "eurlex", "tavily"], '
        '"max_docs": 8, "session_id": "...", "time_range": "day|week|month|year"}'
    )

    def __init__(
        self,
        sgm: SessionGraphManager | None = None,
        ws_broadcaster: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._sgm = sgm
        self._ws_broadcaster = ws_broadcaster

    async def run(self, params: dict) -> dict:
        try:
            query = str(params.get("query", "")).strip()
            if not query:
                return {"error": "query is required", "documents": [], "count": 0}

            max_docs = int(params.get("max_docs", 8) or 8)
            max_docs = max(1, min(max_docs, 20))
            time_range = params.get("time_range")

            source_ids = params.get("sources") or route_sources(query)
            if isinstance(source_ids, str):
                source_ids = [source_ids]
            source_ids = [str(s).strip() for s in source_ids if str(s).strip()]
            if not source_ids:
                source_ids = route_sources(query)

            # Split: Tavily handled separately, EU adapters handled by existing path
            use_tavily = "tavily" in source_ids or os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API")
            eu_source_ids = [s for s in source_ids if s != "tavily"][:3]

            # Build tasks — EU adapters (up to 3) + Tavily in parallel
            all_tasks: list[Any] = []
            all_labels: list[str] = []

            for source_id in eu_source_ids:
                all_tasks.append(
                    asyncio.wait_for(
                        self._search_single_source(source_id, query, max_docs),
                        timeout=15,
                    )
                )
                all_labels.append(source_id)

            if use_tavily:
                from agent.tools.tavily_tool import TavilySearchTool
                tavily = TavilySearchTool()
                all_tasks.append(
                    asyncio.wait_for(
                        tavily.run({"query": query, "max_results": max_docs, "time_range": time_range}),
                        timeout=20,
                    )
                )
                all_labels.append("tavily")

            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            documents: list[dict[str, Any]] = []
            for label, result in zip(all_labels, results):
                if isinstance(result, Exception):
                    log.warning("Search source timed out/failed (%s): %s", label, result)
                    continue
                # Tavily returns {documents: [...]} same as EU adapter post-processing
                if isinstance(result, dict) and "documents" in result:
                    documents.extend(result["documents"])
                elif isinstance(result, list):
                    documents.extend(result)

            log.info("Search merged %d documents from %s", len(documents), all_labels)
            ranked = rank_documents(query, documents, top_k=min(max_docs, 12))

            session_id = str(params.get("session_id", "")).strip()
            if session_id and self._sgm and documents:
                from agent.tools.graphgen_tool import GraphGenTool

                domain = _detect_domain_from_sources(eu_source_ids)
                graphgen = GraphGenTool(sgm=self._sgm, ws_broadcaster=self._ws_broadcaster)
                task = asyncio.create_task(
                    graphgen.run(
                        {
                            "documents": documents,
                            "session_id": session_id,
                            "ontology_domain": domain,
                        }
                    )
                )
                task.add_done_callback(_log_background_graphgen_failure)

            return {
                "documents": ranked,
                "count": len(ranked),
                "total_fetched": len(documents),
            }
        except Exception as exc:
            log.exception("search tool failed")
            return {"error": str(exc), "documents": [], "count": 0}

    async def _search_single_source(self, source_id: str, query: str, max_docs: int) -> list[dict[str, Any]]:
        adapter = self._build_adapter(source_id)
        if adapter is None:
            log.warning("Unknown source adapter: %s", source_id)
            return []

        if isinstance(adapter, SourceAdapter):
            return await self._search_source_adapter(adapter, query, max_docs)
        return await self._search_base_source(adapter, query, max_docs, source_id)

    async def _search_source_adapter(self, adapter: SourceAdapter, query: str, max_docs: int) -> list[dict[str, Any]]:
        search_results = await adapter.search(SearchQuery(query=query, limit=max_docs))
        search_results = search_results[:max_docs]

        async def _fetch_document(result: Any) -> dict[str, Any] | None:
            try:
                doc = await adapter.fetch(result.id)
                return {
                    "id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "url": result.url or doc.metadata.get("url", ""),
                    "source": doc.source,
                    "date": result.date or doc.metadata.get("published") or doc.metadata.get("session_date") or doc.metadata.get("updated") or "",
                }
            except Exception as exc:
                log.warning("Adapter fetch failed (%s:%s): %s", adapter.source_id, getattr(result, "id", ""), exc)
                return None

        fetched = await asyncio.gather(*[_fetch_document(r) for r in search_results], return_exceptions=True)
        docs: list[dict[str, Any]] = []
        for item in fetched:
            if isinstance(item, Exception):
                continue
            if item is not None:
                docs.append(item)
        return docs

    async def _search_base_source(self, adapter: Any, query: str, max_docs: int, source_id: str) -> list[dict[str, Any]]:
        async with adapter:
            parsed_docs = await adapter.scrape(limit=max_docs)

            q = query.lower()
            filtered = [
                d for d in parsed_docs
                if q in (d.title or "").lower() or q in (d.raw_text or "").lower() or q in (d.markdown or "").lower()
            ]
            selected = filtered[:max_docs]

            async def _fetch_one(doc: Any) -> dict[str, Any] | None:
                try:
                    fetched = await adapter.fetch_one(doc.url)
                    if fetched is None:
                        fetched = doc
                    return {
                        "id": fetched.doc_id,
                        "title": fetched.title,
                        "text": fetched.raw_text or fetched.markdown,
                        "url": fetched.url,
                        "source": source_id,
                        "date": fetched.date,
                    }
                except Exception as exc:
                    log.warning("Base source fetch failed (%s:%s): %s", source_id, getattr(doc, "url", ""), exc)
                    return None

            fetched_docs = await asyncio.gather(*[_fetch_one(d) for d in selected], return_exceptions=True)
            docs: list[dict[str, Any]] = []
            for item in fetched_docs:
                if isinstance(item, Exception):
                    continue
                if item is not None:
                    docs.append(item)
            return docs

    def _build_adapter(self, source_id: str) -> Any | None:
        registry = {
            "ep_debates": EPDebatesAdapter,
            "ep_resolutions": EPResolutionsSource,
            "eurlex": EurLexAdapter,
            "council_conclusions": CouncilConclusionsSource,
            "ec_press": ECPressSource,
            "ecb_publications": ECBPublicationsSource,
            "eca_reports": ECAReportsSource,
            "eprs_briefings": EPRSBriefingsSource,
            "cellar_sparql": CellarSPARQLSource,
        }
        adapter_cls = registry.get(source_id)
        return adapter_cls() if adapter_cls else None


def _detect_domain_from_sources(source_ids: list[str]) -> str:
    sources = {str(source_id).lower() for source_id in source_ids}
    if "ep_debates" in sources:
        return "debates"
    if "eurlex" in sources or "cellar_sparql" in sources or "ep_resolutions" in sources:
        return "legislation"
    if sources & {"ecb_publications", "council_conclusions", "eca_reports"}:
        return "finance"
    return "general"


def _log_background_graphgen_failure(task: asyncio.Task[dict[str, Any]]) -> None:
    if task.cancelled():
        return
    try:
        result = task.result()
        if isinstance(result, dict) and result.get("error"):
            log.warning("Background graphgen returned error: %s", result.get("error"))
    except Exception as exc:
        log.warning("Background graphgen failed: %s", exc)
