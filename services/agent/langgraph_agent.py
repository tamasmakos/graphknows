"""LangGraph-based EU investigator agent for Graphknows."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Awaitable, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage  # noqa: F401
from langchain_core.tools import tool

log = logging.getLogger("agent.langgraph_agent")

# ── AgentState — module-level so Python 3.13 get_type_hints can resolve it ──
class AgentState(TypedDict):
    """Shared state passed through the LangGraph agent graph."""
    messages: Annotated[list, add_messages]

# ── Tool definitions ──────────────────────────────────────────────────────────

# Broadcaster closure — injected by build_graph()
_ws_broadcaster: Callable[[str, dict], Awaitable[None]] | None = None
_session_id: str = ""


async def _noop_broadcaster(session_id: str, payload: dict) -> None:
    pass


# ── Lexical graph builder ────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "is", "are",
    "was", "were", "on", "at", "by", "with", "this", "that", "as", "be",
    "has", "have", "its", "it", "from", "not", "but", "their", "they",
    "which", "who", "will", "been", "more", "about", "also", "after",
    "before", "than", "into", "through", "during", "between", "against",
    "each", "other", "those", "these", "such", "when", "where", "would",
})


def _build_lexical_graph(documents: list[dict]) -> dict:
    """Build a simple term co-occurrence graph from document titles/snippets."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_terms: set[str] = set()

    for i, doc in enumerate(documents[:15]):
        doc_id = doc.get("id", f"doc-{i}")
        doc_title = doc.get("title", "")
        text = f"{doc_title} {doc.get('text', '')}"[:2000]

        # Extract capitalized terms / noun phrases
        raw_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b', text)
        terms = [t for t in raw_terms if t.lower() not in _STOPWORDS and len(t) > 3]
        unique_terms = list(dict.fromkeys(terms))[:15]

        # Add doc node
        doc_node_id = f"DOC::{doc_id}"
        if doc_node_id not in seen_terms:
            nodes.append({
                "id": doc_node_id,
                "type": "Document",
                "name": doc_title[:60] or doc_id,
                "doc_id": doc_id,
            })
            seen_terms.add(doc_node_id)

        for term in unique_terms:
            if term not in seen_terms:
                nodes.append({"id": term, "type": "CONCEPT", "name": term})
                seen_terms.add(term)
            edges.append({"source": doc_node_id, "target": term, "type": "MENTIONS"})

    return {"nodes": nodes, "edges": edges}


# ── Graphgen subprocess ───────────────────────────────────────────────────────

async def _run_graphgen(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
) -> tuple[list[dict], list[dict], str]:
    """Run graphgen subprocess or keyword fallback. Returns (nodes, edges, summary_str)."""
    import psutil  # type: ignore

    lexical = _build_lexical_graph(documents)

    try:
        mem = psutil.virtual_memory()
        if mem.available < 2.0 * 1024 ** 3:
            log.warning("Low RAM — using keyword graph fallback")
            return await _keyword_fallback(documents, session_id, broadcaster, lexical)
    except Exception:
        pass

    worker = Path(__file__).parent / "tools" / "graphgen_worker.py"
    payload = json.dumps({
        "documents": documents[:5],
        "session_id": session_id,
        "ontology_domain": "general",
    }).encode()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=180)
        if proc.returncode != 0:
            log.warning("Graphgen worker failed: %s", stderr.decode()[:200])
            return await _keyword_fallback(documents, session_id, broadcaster, lexical)

        result = json.loads(stdout)
        nodes = result.get("entities", [])[:200]
        edges = result.get("relations", [])[:500]
        summary = f"Graph built: {len(nodes)} nodes, {len(edges)} edges"

        try:
            await broadcaster(session_id, {
                "type": "graph_update",
                "graph": {"nodes": nodes, "edges": edges},
                "lexical": lexical,
            })
        except Exception:
            pass

        return nodes, edges, summary

    except asyncio.TimeoutError:
        log.warning("Graphgen timed out")
        return await _keyword_fallback(documents, session_id, broadcaster, lexical)
    except Exception as exc:
        log.warning("Graphgen error: %s", exc)
        return await _keyword_fallback(documents, session_id, broadcaster, lexical)


async def _keyword_fallback(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
    lexical: dict,
) -> tuple[list[dict], list[dict], str]:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    for doc in documents[:12]:
        text = str(doc.get("text", "") or doc.get("title", ""))[:2000]
        terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        unique = list(dict.fromkeys(t for t in terms if t.lower() not in _STOPWORDS and len(t) > 3))[:15]
        for term in unique:
            if term not in seen:
                nodes.append({"id": term, "type": "CONCEPT", "name": term})
                seen.add(term)
        for i, a in enumerate(unique):
            for b in unique[i + 1: i + 4]:
                edges.append({"source": a, "target": b, "type": "CO_OCCURS"})

    try:
        await broadcaster(session_id, {
            "type": "graph_update",
            "graph": {"nodes": nodes, "edges": edges},
            "lexical": lexical,
        })
    except Exception:
        pass

    return nodes, edges, f"Graph built: {len(nodes)} nodes, {len(edges)} edges (keyword fallback)"


# ── LangGraph agent factory ──────────────────────────────────────────────────

def build_graph(
    ws_broadcaster: Callable[[str, dict], Awaitable[None]] | None,
    session_id: str,
    groq_api_key: str,
):
    """Build and return a compiled LangGraph agent graph."""
    broadcaster = ws_broadcaster or _noop_broadcaster

    # ── Tool 1: search_web ───────────────────────────────────────────────────
    @tool
    async def search_web(query: str, max_results: int = 8) -> str:
        """Search the live web for news, context, and general information.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (default 8).
        """
        try:
            from tavily import AsyncTavilyClient
            api_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_API", "")
            if not api_key:
                return "Error: TAVILY_API_KEY not configured"
            client = AsyncTavilyClient(api_key=api_key)
            resp = await asyncio.wait_for(
                client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced",
                    include_raw_content=True,
                    include_answer=False,
                ),
                timeout=25,
            )
            results = resp.get("results", [])
            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = (r.get("raw_content") or r.get("content") or "")[:300]
                date = r.get("published_date", "")
                lines.append(f"{i}. {title}\n   URL: {url}\n   Date: {date}\n   {snippet}")
            return "\n\n".join(lines) if lines else "No results found."
        except Exception as exc:
            log.warning("search_web failed: %s", exc)
            return f"Search failed: {exc}"

    # ── Tool 2: search_eu ────────────────────────────────────────────────────
    @tool
    async def search_eu(
        query: str,
        sources: str = "ep_debates,eurlex,ec_press",
        max_docs: int = 8,
    ) -> str:
        """Search EU official archives: EP debates, EUR-Lex, Council, EC press releases.

        Args:
            query: Search query string.
            sources: Comma-separated source IDs (ep_debates, eurlex, ec_press, etc.).
            max_docs: Maximum documents to return (default 8).
        """
        _SERVICES_DIR = Path(__file__).resolve().parents[1]
        if str(_SERVICES_DIR) not in sys.path:
            sys.path.insert(0, str(_SERVICES_DIR))

        from search.models import SearchQuery
        from search.adapters.base import SourceAdapter

        ADAPTER_REGISTRY: dict[str, Any] = {}
        try:
            from search.adapters.ep_debates import EPDebatesAdapter
            ADAPTER_REGISTRY["ep_debates"] = EPDebatesAdapter
        except Exception:
            pass
        try:
            from search.adapters.eurlex import EurLexAdapter
            ADAPTER_REGISTRY["eurlex"] = EurLexAdapter
        except Exception:
            pass
        try:
            from search.adapters.ec_press import ECPressSource
            ADAPTER_REGISTRY["ec_press"] = ECPressSource
        except Exception:
            pass
        try:
            from search.adapters.ep_resolutions import EPResolutionsSource
            ADAPTER_REGISTRY["ep_resolutions"] = EPResolutionsSource
        except Exception:
            pass
        try:
            from search.adapters.council_conclusions import CouncilConclusionsSource
            ADAPTER_REGISTRY["council_conclusions"] = CouncilConclusionsSource
        except Exception:
            pass
        try:
            from search.adapters.ecb_publications import ECBPublicationsSource
            ADAPTER_REGISTRY["ecb_publications"] = ECBPublicationsSource
        except Exception:
            pass
        try:
            from search.adapters.eprs_briefings import EPRSBriefingsSource
            ADAPTER_REGISTRY["eprs_briefings"] = EPRSBriefingsSource
        except Exception:
            pass

        source_ids = [s.strip() for s in sources.split(",") if s.strip()]
        documents: list[dict] = []

        async def _search_one(sid: str) -> list[dict]:
            cls = ADAPTER_REGISTRY.get(sid)
            if not cls:
                return []
            adapter = cls()
            try:
                if isinstance(adapter, SourceAdapter):
                    sq = SearchQuery(query=query, limit=max_docs)
                    results = await asyncio.wait_for(adapter.search(sq), timeout=15)
                    docs = []
                    for r in results[:max_docs]:
                        try:
                            doc = await asyncio.wait_for(adapter.fetch(r.id), timeout=10)
                            docs.append({
                                "id": doc.id,
                                "title": doc.title,
                                "text": doc.text,
                                "url": getattr(r, "url", "") or doc.metadata.get("url", ""),
                                "source": sid,
                                "date": getattr(r, "date", "") or doc.metadata.get("published", ""),
                            })
                        except Exception:
                            pass
                    return docs
                else:
                    # BaseScraper style
                    async with adapter:
                        parsed = await asyncio.wait_for(adapter.scrape(limit=max_docs), timeout=15)
                        q = query.lower()
                        filtered = [d for d in parsed if q in (d.title or "").lower() or q in (d.raw_text or "").lower()][:max_docs]
                        docs = []
                        for d in filtered:
                            docs.append({
                                "id": getattr(d, "doc_id", d.url),
                                "title": d.title,
                                "text": d.raw_text or d.markdown or "",
                                "url": d.url,
                                "source": sid,
                                "date": getattr(d, "date", ""),
                            })
                        return docs
            except Exception as exc:
                log.warning("search_eu adapter %s failed: %s", sid, exc)
                return []

        tasks = [_search_one(sid) for sid in source_ids[:4]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                documents.extend(r)

        if not documents:
            return "No EU documents found for this query."

        lines = []
        for i, doc in enumerate(documents[:max_docs], 1):
            lines.append(
                f"{i}. {doc.get('title', 'Untitled')}\n"
                f"   Institution: {doc.get('source', '')}\n"
                f"   Date: {doc.get('date', '')}\n"
                f"   URL: {doc.get('url', '')}\n"
                f"   {str(doc.get('text', ''))[:300]}"
            )
        return "\n\n".join(lines)

    # ── Tool 3: graphgen ─────────────────────────────────────────────────────
    @tool
    async def graphgen(session_id_arg: str, documents_json: str) -> str:
        """Build a knowledge graph from collected documents.

        Args:
            session_id_arg: The session ID for this investigation.
            documents_json: JSON string of document list (title, text, url, source, date).
        """
        try:
            docs = json.loads(documents_json)
        except Exception as exc:
            return f"Error parsing documents JSON: {exc}"

        if not docs:
            return "No documents provided for graph generation."

        _, _, summary = await _run_graphgen(docs, session_id_arg, broadcaster)
        return summary

    # ── Tool 4: graphrag ─────────────────────────────────────────────────────
    @tool
    async def graphrag(query: str, session_id_arg: str, k: int = 5) -> str:
        """Query the GraphRAG service for deeper entity analysis from the knowledge graph.

        Args:
            query: The query to send to GraphRAG.
            session_id_arg: Session ID for context.
            k: Number of results to retrieve.
        """
        import httpx
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "http://localhost:8001/retrieve",
                    json={"query": query, "session_id": session_id_arg, "max_steps": k},
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "")
                entities = data.get("entities_mentioned", [])
                if entities:
                    entity_str = ", ".join(str(e) if not isinstance(e, dict) else e.get("name", "") for e in entities[:10])
                    return f"Entities: {entity_str}\n\nAnswer: {answer}"
                return answer or "No results from GraphRAG."
        except Exception as exc:
            return f"GraphRAG unavailable: {exc}"

    # ── System prompt ────────────────────────────────────────────────────────
    from datetime import date as _date
    today = _date.today().isoformat()

    SYSTEM_PROMPT = f"""You are an EU public records investigator for the European Parliamentary Research Service.
Today's date: {today}

STRICT RESEARCH PROTOCOL:
1. Call search_eu to find official EU records (EP debates, EUR-Lex, Council decisions, EC press releases)
2. Call search_web for live news, context, academic analysis
3. Call graphgen to build a knowledge graph from the collected documents (pass all docs as JSON)
4. Optionally call graphrag for deeper entity analysis from the accumulated KG
5. Write a full EPRS-style investigative report as your final response

DO NOT loop. Max 2 search calls, 1 graphgen call, 1 graphrag call, then write your final ANSWER.

When calling graphgen, construct the documents_json parameter as a JSON array of objects with keys: title, text, url, source, date.
The session_id_arg for graphgen and graphrag is: {session_id}

REPORT FORMAT (use this exact structure):
## [Investigation Title]
**European Parliamentary Research Service** | {today}

### Executive Summary
[3-4 sentences, key finding, confidence]

### Background
[Policy context]

### Key Findings
1. [Finding] — [🏛️ Source title](url)
2. [Finding] — [🌐 Source](url)

### Legislative Record
- [CELEX if available] — [title] — [institution] — [date]

### Context & Analysis
[web sources, commentary]

### Key Entities
**People:** ... **Organizations:** ... **Legislation:** ...

### Sources
- [🏛️ Doc-1] Full citation — institution — date — url
- [🌐 Doc-2] ...
"""

    # ── State & graph ────────────────────────────────────────────────────────
    # AgentState defined at module level (Python 3.13 get_type_hints compat)

    tools_list = [search_web, search_eu, graphgen, graphrag]

    llm_base = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        streaming=True,
    )
    # Bind tools — force tool use on first call to prevent direct answer
    llm_with_tools = llm_base.bind_tools(tools_list)
    llm_force_tools = llm_base.bind_tools(tools_list, tool_choice="required")

    tool_node = ToolNode(tools_list)

    async def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        # Prepend system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        # On first turn (only 2 messages: system + human), force tool use
        human_msgs = [m for m in messages if getattr(m, "type", None) == "human"]
        tool_msgs = [m for m in messages if getattr(m, "type", None) == "tool"]
        force = len(tool_msgs) == 0  # No tool results yet → force a search
        active_llm = llm_force_tools if force else llm_with_tools
        response = await active_llm.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state):  # no type hints — LangGraph introspects with get_type_hints
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
