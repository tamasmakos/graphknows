"""Search tool — queries EU public archives + Tavily web search, merges and ranks results."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

from agent.tools.base import BaseTool
from agent.tools.source_router import route_sources
from agent.tools.search_client import SearchServiceClient
from agent.session.manager import SessionGraphManager
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
                domain = _detect_domain_from_sources(eu_source_ids)
                task = asyncio.create_task(
                    _run_graphgen_subprocess(
                        documents=documents,
                        session_id=session_id,
                        domain=domain,
                        sgm=self._sgm,
                        ws_broadcaster=self._ws_broadcaster,
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
        """Fetch documents for one source via the search service HTTP API.

        All source adapter logic lives in the search service.  The agent never
        imports adapters directly — it stays on the right side of the service
        boundary.
        """
        client = SearchServiceClient()
        try:
            raw_docs = await client.fetch_documents(
                query=query,
                sources=[source_id],
                limit=max_docs,
            )
        except Exception as exc:
            log.warning("SearchServiceClient failed (%s): %s", source_id, exc)
            return []
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "url": doc.url,
                "source": doc.source,
                "date": doc.date or "",
            }
            for doc in raw_docs
        ]


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


async def _run_graphgen_subprocess(
    documents: list[dict[str, Any]],
    session_id: str,
    domain: str,
    sgm: Any,
    ws_broadcaster: Any,
) -> dict[str, Any]:
    """Run GLiNER2 graphgen in an isolated subprocess so it cannot crash the agent.

    Skips if available RAM < 2 GB to avoid OOM on the host.
    """
    import json as _json
    import psutil  # type: ignore

    # RAM guard — GLiNER2 needs ~2 GB; skip if too tight
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        if available_gb < 2.0:
            log.warning("Skipping graphgen — only %.1f GB RAM available (need 2 GB)", available_gb)
            # Fall back to lightweight keyword co-occurrence graph
            return await _run_keyword_graph(documents, session_id, sgm, ws_broadcaster)
    except Exception:
        pass

    worker = Path(__file__).parent / "graphgen_worker.py"
    payload = _json.dumps({
        "documents": documents[:5],  # cap at 5 docs to limit RAM pressure
        "session_id": session_id,
        "ontology_domain": domain,
    }).encode()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=180)
        if stderr:
            log.debug("Graphgen worker stderr: %s", stderr.decode()[:500])
        if proc.returncode != 0:
            log.warning("Graphgen worker exited %d: %s", proc.returncode, stderr.decode()[:200])
            return await _run_keyword_graph(documents, session_id, sgm, ws_broadcaster)

        result = _json.loads(stdout)
        nodes_data = result.get("entities", [])
        edges_data = result.get("relations", [])

        log.info("Graphgen subprocess: %d nodes, %d edges for session %s",
                 len(nodes_data), len(edges_data), session_id)

        # Persist to session graph manager
        if nodes_data and sgm:
            try:
                for node in nodes_data:
                    await sgm.add_entity(session_id, node.get("id", ""), node.get("type", "CONCEPT"), {})
                for edge in edges_data:
                    await sgm.add_relation(
                        session_id,
                        edge.get("source", ""), edge.get("target", ""), edge.get("type", "co-occurs"), {}
                    )
            except Exception as exc:
                log.warning("Failed to persist graph to session manager: %s", exc)

        if ws_broadcaster:
            try:
                await ws_broadcaster(session_id, {
                    "type": "graph_update",
                    "graph": {
                        "nodes": nodes_data[:200],
                        "edges": edges_data[:500],
                    },
                })
            except Exception:
                pass

        return result

    except asyncio.TimeoutError:
        log.warning("Graphgen subprocess timed out for session %s", session_id)
        return await _run_keyword_graph(documents, session_id, sgm, ws_broadcaster)
    except Exception as exc:
        log.warning("Graphgen subprocess error: %s", exc)
        return {"error": str(exc), "nodes": 0, "edges": 0}


async def _run_keyword_graph(
    documents: list[dict[str, Any]],
    session_id: str,
    sgm: Any,
    ws_broadcaster: Any,
) -> dict[str, Any]:
    """Lightweight fallback: build a co-occurrence graph from noun phrases without ML models."""
    import re as _re

    _STOPWORDS = frozenset({"the", "a", "an", "of", "in", "to", "and", "or", "for",
                            "is", "are", "was", "were", "on", "at", "by", "with",
                            "this", "that", "as", "be", "has", "have", "its", "it"})

    def _extract_terms(text: str) -> list[str]:
        words = _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return [w for w in words if w.lower() not in _STOPWORDS and len(w) > 3]

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()

    for doc in documents[:12]:
        text = str(doc.get("text", "") or doc.get("title", ""))[:2000]
        terms = _extract_terms(text)
        # Deduplicate
        unique = list(dict.fromkeys(terms))[:15]
        for term in unique:
            if term not in seen_nodes:
                nodes.append({"id": term, "type": "CONCEPT", "doc_id": doc.get("id", ""), "name": term})
                seen_nodes.add(term)
        for i, a in enumerate(unique):
            for b in unique[i + 1: i + 4]:
                edges.append({"source": a, "target": b, "type": "CO_OCCURS",
                               "properties": {"relation_type": "co-occurs"}})

    log.info("Keyword fallback graph: %d nodes, %d edges for session %s",
             len(nodes), len(edges), session_id)

    if nodes and sgm:
        try:
            for node in nodes:
                await sgm.add_entity(session_id, node["id"], node["type"], {})
            for edge in edges:
                await sgm.add_relation(session_id, edge["source"], edge["target"], edge["type"], {})
        except Exception as exc:
            log.warning("Failed to persist keyword graph: %s", exc)

    if ws_broadcaster:
        try:
            await ws_broadcaster(session_id, {
                "type": "graph_update",
                "graph": {"nodes": nodes, "edges": edges},
            })
        except Exception:
            pass

    return {"nodes": len(nodes), "edges": len(edges), "entities": nodes, "relations": edges}
