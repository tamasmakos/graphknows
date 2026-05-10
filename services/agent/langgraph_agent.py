"""LangGraph-based EU investigator agent for Graphknows.

Architecture (wiki: concepts/langgraph-agent-architecture-2026-05-08.md):
- 4 LLM tools: search_web, search_eu, graphgen, graphrag
- GLiNER2 + DSPy entity-relation extraction (graphgen_worker.py subprocess)
- Streaming partial graphs: keyword graph fires immediately after each search;
  GLiNER2/DSPy full graph fires when graphgen tool is called
- Session doc store (_session_docs) accumulates all retrieved documents
- AgentState carries extracted_entities so LLM reasons from the graph
"""
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


# ── Module-level singletons — set by build_graph() per call ──────────────────
_ws_broadcaster: Callable[[str, dict], Awaitable[None]] | None = None
_session_id: str = ""

# Session-scoped document accumulator: session_id → list of collected docs
_session_docs: dict[str, list[dict]] = {}

# Session-scoped extracted entities (from GLiNER2): session_id → entity names
_session_entities: dict[str, list[str]] = {}


async def _noop_broadcaster(session_id: str, payload: dict) -> None:
    pass


# ── Stopwords ────────────────────────────────────────────────────────────────
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "or", "for", "is", "are",
    "was", "were", "on", "at", "by", "with", "this", "that", "as", "be",
    "has", "have", "its", "it", "from", "not", "but", "their", "they",
    "which", "who", "will", "been", "more", "about", "also", "after",
    "before", "than", "into", "through", "during", "between", "against",
    "each", "other", "those", "these", "such", "when", "where", "would",
})


# ── Lexical graph builder (source-anchored) ───────────────────────────────────
def _build_lexical_graph(documents: list[dict]) -> dict:
    """Build a source-anchored term graph: doc nodes linked to key terms."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_terms: set[str] = set()

    for i, doc in enumerate(documents[:15]):
        doc_id = doc.get("id", f"doc-{i}")
        doc_title = doc.get("title", "") or doc_id
        text = f"{doc_title} {doc.get('text', '')}".strip()[:2000]

        raw_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b', text)
        terms = [t for t in raw_terms if t.lower() not in _STOPWORDS and len(t) > 3]
        unique_terms = list(dict.fromkeys(terms))[:15]

        doc_node_id = f"DOC::{doc_id}"
        if doc_node_id not in seen_terms:
            nodes.append({
                "id": doc_node_id,
                "type": "Document",
                "name": doc_title[:60],
                "doc_id": doc_id,
                "url": doc.get("url", ""),
            })
            seen_terms.add(doc_node_id)

        for term in unique_terms:
            if term not in seen_terms:
                nodes.append({"id": term, "type": "CONCEPT", "name": term})
                seen_terms.add(term)
            edges.append({"source": doc_node_id, "target": term, "type": "MENTIONS"})

    return {"nodes": nodes, "edges": edges}


# ── Keyword-only partial graph (fires immediately, no model needed) ──────────
async def _emit_keyword_graph(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
) -> list[str]:
    """Build + broadcast a fast keyword co-occurrence graph. Returns entity names."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()
    entity_names: list[str] = []

    for doc in documents[:12]:
        text = str(doc.get("text", "") or doc.get("title", ""))[:2000]
        title_terms = re.findall(r'\b[A-Z][a-zA-Z]+\b', doc.get("title", ""))
        proper_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        all_terms = title_terms + proper_terms + acronyms
        unique = list(dict.fromkeys(
            t for t in all_terms if t.lower() not in _STOPWORDS and len(t) > 2
        ))[:15]

        for term in unique:
            if term not in seen:
                nodes.append({"id": term, "type": "CONCEPT", "name": term})
                seen.add(term)
                entity_names.append(term)
        for i, a in enumerate(unique):
            for b in unique[i + 1: i + 4]:
                edges.append({"source": a, "target": b, "type": "CO_OCCURS"})

    lexical = _build_lexical_graph(documents)
    try:
        await broadcaster(session_id, {
            "type": "graph_update",
            "graph": {"nodes": nodes, "edges": edges},
            "lexical": lexical,
            "partial": True,  # signal: GLiNER2 graph is still running
        })
    except Exception:
        pass

    return entity_names


# ── GLiNER2 + DSPy subprocess graphgen ───────────────────────────────────────
async def _run_graphgen(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
) -> tuple[list[dict], list[dict], str]:
    """Run GLiNER2+DSPy graphgen subprocess. Returns (nodes, edges, summary_str)."""
    import psutil  # type: ignore

    lexical = _build_lexical_graph(documents)

    # Check available RAM — GLiNER2 needs ~1.5GB
    try:
        mem_avail = psutil.virtual_memory().available
    except Exception:
        mem_avail = 4 * 1024 ** 3  # assume 4GB if psutil fails

    if mem_avail < 1.5 * 1024 ** 3:
        log.warning("Low RAM (%.1fGB) — skipping GLiNER2, using keyword graph", mem_avail / 1024**3)
        return await _keyword_fallback_graphgen(documents, session_id, broadcaster, lexical)

    worker = Path(__file__).parent / "tools" / "graphgen_worker.py"
    payload = json.dumps({
        "documents": [
            {
                "id": d.get("id", d.get("url", "")[:60]),
                "title": d.get("title", ""),
                "text": str(d.get("text", ""))[:2000],
            }
            for d in documents[:10]
        ],
        "session_id": session_id,
        "ontology_domain": "eu_politics",
    }).encode()

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(worker),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=240)

        if proc.returncode != 0 or not stdout.strip():
            err_msg = stderr.decode()[:300] if stderr else "empty stdout"
            log.warning("Graphgen worker failed (rc=%d): %s", proc.returncode, err_msg)
            return await _keyword_fallback_graphgen(documents, session_id, broadcaster, lexical)

        result = json.loads(stdout)
        nodes = result.get("entities", [])[:300]
        edges = result.get("relations", [])[:600]
        n_nodes, n_edges = len(nodes), len(edges)
        summary = f"Graph built: {n_nodes} entities, {n_edges} relations (GLiNER2 + DSPy)"

        # Store extracted entity names for LLM reasoning
        entity_names = list(dict.fromkeys(
            node.get("name", node.get("id", "")) for node in nodes
        ))[:50]
        _session_entities.setdefault(session_id, [])
        existing = set(_session_entities[session_id])
        _session_entities[session_id].extend(e for e in entity_names if e not in existing)

        try:
            await broadcaster(session_id, {
                "type": "graph_update",
                "graph": {"nodes": nodes, "edges": edges},
                "lexical": lexical,
                "partial": False,
            })
        except Exception:
            pass

        log.info("Graph update broadcast: %d nodes, %d edges for %s", n_nodes, n_edges, session_id)
        return nodes, edges, summary

    except asyncio.TimeoutError:
        log.warning("Graphgen subprocess timed out after 180s")
        return await _keyword_fallback_graphgen(documents, session_id, broadcaster, lexical)
    except Exception as exc:
        log.warning("Graphgen error: %s", exc)
        return await _keyword_fallback_graphgen(documents, session_id, broadcaster, lexical)


async def _keyword_fallback_graphgen(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
    lexical: dict,
) -> tuple[list[dict], list[dict], str]:
    """Co-occurrence keyword graph as final fallback when GLiNER2 fails."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()

    for doc in documents[:12]:
        text = str(doc.get("text", "") or doc.get("title", ""))[:2000]
        title_terms = re.findall(r'\b[A-Z][a-zA-Z]+\b', doc.get("title", ""))
        proper_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        unique = list(dict.fromkeys(
            t for t in (title_terms + proper_terms + acronyms)
            if t.lower() not in _STOPWORDS and len(t) > 2
        ))[:15]
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
            "partial": False,
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

        After results are returned, a partial knowledge graph is automatically
        streamed to the UI — you do NOT need to call graphgen manually yet.

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
            docs_to_store = []
            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = (r.get("raw_content") or r.get("content") or "")[:400]
                date = r.get("published_date", "")
                lines.append(f"{i}. {title}\n   URL: {url}\n   Date: {date}\n   {snippet}")
                docs_to_store.append({
                    "id": url[:80],
                    "title": title,
                    "url": url,
                    "text": snippet,
                    "date": date,
                    "source": "web",
                })

            _session_docs.setdefault(session_id, []).extend(docs_to_store)

            # Fire partial keyword graph immediately — non-blocking background task
            if docs_to_store:
                asyncio.create_task(
                    _emit_keyword_graph(
                        _session_docs.get(session_id, []),
                        session_id,
                        broadcaster,
                    )
                )

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
        """Search EU official public archives: EP debates, EUR-Lex, Council, EC press releases.

        After results are returned, a partial knowledge graph is automatically
        streamed to the UI — you do NOT need to call graphgen manually yet.

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
        for mod_name, cls_name, key in [
            ("search.adapters.ep_debates", "EPDebatesAdapter", "ep_debates"),
            ("search.adapters.eurlex", "EurLexAdapter", "eurlex"),
            ("search.adapters.ec_press", "ECPressSource", "ec_press"),
            ("search.adapters.ep_resolutions", "EPResolutionsSource", "ep_resolutions"),
            ("search.adapters.council_conclusions", "CouncilConclusionsSource", "council_conclusions"),
            ("search.adapters.ecb_publications", "ECBPublicationsSource", "ecb_publications"),
            ("search.adapters.eprs_briefings", "EPRSBriefingsSource", "eprs_briefings"),
        ]:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                ADAPTER_REGISTRY[key] = getattr(mod, cls_name)
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
                                "text": str(doc.text or "")[:4000],
                                "url": getattr(r, "url", "") or doc.metadata.get("url", ""),
                                "source": sid,
                                "date": getattr(r, "date", "") or doc.metadata.get("published", ""),
                            })
                        except Exception:
                            pass
                    return docs
                else:
                    async with adapter:
                        parsed = await asyncio.wait_for(adapter.scrape(limit=max_docs), timeout=15)
                        q = query.lower()
                        filtered = [
                            d for d in parsed
                            if q in (d.title or "").lower() or q in (d.raw_text or "").lower()
                        ][:max_docs]
                        return [
                            {
                                "id": getattr(d, "doc_id", d.url),
                                "title": d.title,
                                "text": str(d.raw_text or d.markdown or "")[:4000],
                                "url": d.url,
                                "source": sid,
                                "date": getattr(d, "date", ""),
                            }
                            for d in filtered
                        ]
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

        _session_docs.setdefault(session_id, []).extend(documents[:max_docs])

        # Fire partial keyword graph immediately — non-blocking
        if documents:
            asyncio.create_task(
                _emit_keyword_graph(
                    _session_docs.get(session_id, []),
                    session_id,
                    broadcaster,
                )
            )

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
    async def graphgen(session_id_arg: str) -> str:
        """Run full GLiNER2 + DSPy entity-relation extraction on all collected documents.

        This produces a high-quality knowledge graph with real named entities and
        typed relations (extracted by GLiNER2 ontology-guided NER + DSPy LLM triplets).
        Call this ONCE after both search tools have returned results.

        The extracted entities will be listed in the result so you can reference them
        in your final report.

        Args:
            session_id_arg: The session ID for this investigation.
        """
        docs = _session_docs.get(session_id_arg) or _session_docs.get(session_id, [])
        if not docs:
            await asyncio.sleep(3)
            docs = _session_docs.get(session_id_arg) or _session_docs.get(session_id, [])
        if not docs:
            return "No documents collected yet — run search_eu or search_web first."

        nodes, edges, summary = await _run_graphgen(docs, session_id_arg or session_id, broadcaster)

        # Return entity list so LLM can reason from it
        entities = _session_entities.get(session_id_arg or session_id, [])
        if entities:
            top_entities = ", ".join(entities[:30])
            return (
                f"{summary}\n\n"
                f"**Key entities extracted:** {top_entities}\n\n"
                f"Use these entities in your final report under 'Key Graph Entities'. "
                f"Reference specific people, organizations, and legislation found."
            )
        return summary

    # ── Tool 4: graphrag ─────────────────────────────────────────────────────
    @tool
    async def graphrag(query: str, session_id_arg: str, k: int = 5) -> str:
        """Query the accumulated knowledge graph for entity relationships and patterns.

        Uses the session's extracted entities to answer graph-structure questions.
        Also queries the persistent Neo4j KG for cross-session knowledge.

        Args:
            query: The graph query (e.g. 'who is most connected to Von der Leyen?').
            session_id_arg: Session ID for context.
            k: Number of results to retrieve.
        """
        # First try in-session entity index
        sid = session_id_arg or session_id
        entities = _session_entities.get(sid, [])
        if entities and any(e.lower() in query.lower() for e in entities[:20]):
            matched = [e for e in entities if e.lower() in query.lower()]
            docs = _session_docs.get(sid, [])
            related = [
                d.get("title", "") for d in docs
                if any(m.lower() in (d.get("text", "") + d.get("title", "")).lower()
                       for m in matched[:3])
            ][:5]
            if related:
                return (
                    f"Session graph entities related to '{query}':\n"
                    f"Matched: {', '.join(matched[:5])}\n"
                    f"Supporting documents: {'; '.join(related)}"
                )

        # Fall back to Neo4j GraphRAG service
        import httpx
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    "http://localhost:8001/retrieve",
                    json={"query": query, "session_id": sid, "max_steps": k},
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "")
                entity_list = data.get("entities_mentioned", [])
                if entity_list:
                    entity_str = ", ".join(
                        str(e) if not isinstance(e, dict) else e.get("name", "")
                        for e in entity_list[:10]
                    )
                    return f"Entities: {entity_str}\n\nAnswer: {answer}"
                return answer or "No results from GraphRAG."
        except Exception as exc:
            if entities:
                return (
                    f"GraphRAG service unavailable. Session entities so far: "
                    f"{', '.join(entities[:20])}"
                )
            return f"GraphRAG unavailable: {exc}"

    # ── System prompt ────────────────────────────────────────────────────────
    from datetime import date as _date
    today = _date.today().isoformat()

    SYSTEM_PROMPT = f"""You are an EU public records investigator for the European Parliamentary Research Service.
Today's date: {today}
Session ID: {session_id}

RESEARCH PROTOCOL — follow this EXACT sequence:

STEP 1 (one LLM response): Call search_eu AND search_web simultaneously.
  → A partial keyword graph is automatically streamed to the user as you search.

STEP 2 (next LLM response): Call graphgen(session_id_arg="{session_id}")
  → This runs GLiNER2 + DSPy to extract real named entities and typed relations.
  → The result includes the key entities found — use them in your report.
  → CRITICAL: Call graphgen in a SEPARATE response AFTER receiving search results.

STEP 3 (optional): Call graphrag if the user needs entity connectivity analysis.

STEP 4: Write your final EPRS-style investigative report.

RULES:
- Do NOT call graphgen in the same response as search tools (they run in parallel, docs won't be ready)
- Do NOT loop or re-search without a clear reason
- The report MUST include a 'Key Graph Entities' section listing the entities from graphgen
- Cite sources inline with [🏛️ Source] for EU institutions, [🌐 Source] for web

REPORT FORMAT:
## [Investigation Title]
**EPRS Intelligence Brief** | Session {session_id[:8]} | {today}

### Executive Summary
[2-3 sentences, key finding, confidence level]

### Key Findings
1. [Finding] — [🏛️ Source](url)
2. [Finding] — [🌐 Source](url)

### Legislative Record
- [CELEX/ELI if available] — [title] — [institution] — [date]

### Context & Analysis
[synthesis across sources, tensions, open questions]

### Key Graph Entities
**People:** [list from graphgen result]
**Organizations:** [list from graphgen result]
**Legislation & Policies:** [list from graphgen result]

### Sources
- [🏛️ Doc-1] Full citation — institution — date — url
- [🌐 Doc-2] ...
"""

    # ── Build graph ──────────────────────────────────────────────────────────
    tools_list = [search_web, search_eu, graphgen, graphrag]

    llm_base = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        streaming=True,
    )
    llm_with_tools = llm_base.bind_tools(tools_list)
    llm_force_tools = llm_base.bind_tools(tools_list, tool_choice="required")

    tool_node = ToolNode(tools_list)

    async def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        tool_msgs = [m for m in messages if getattr(m, "type", None) == "tool"]
        force = len(tool_msgs) == 0
        active_llm = llm_force_tools if force else llm_with_tools
        response = await active_llm.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state):  # no type hints — Python 3.13 get_type_hints compat
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
