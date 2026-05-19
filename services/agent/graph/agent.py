"""LangGraph-based EU investigator agent for Graphknows.

Architecture (wiki: concepts/langgraph-agent-architecture-2026-05-08.md):
- 4 LLM tools: search_web, search_eu, graphgen, hybridrag
- GLiNER2 + DSPy entity-relation extraction (InProcessExtractor)
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
from langchain_openai import ChatOpenAI
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


def clear_session_state(session_id: str) -> None:
    """Remove accumulated docs and entities for a completed session.

    Call this from the session end handler to prevent unbounded memory growth.

    Args:
        session_id: The session to clear from the in-process accumulators.
    """
    _session_docs.pop(session_id, None)
    _session_entities.pop(session_id, None)


async def _noop_broadcaster(session_id: str, payload: dict) -> None:
    pass  # NOSONAR — intentional no-op; replaced by caller-supplied ws_broadcaster


try:
    from services.agent.graph.workflow import REQUIRED_STREAM_EVENTS as _REQUIRED_STREAM_EVENTS
except Exception:
    _REQUIRED_STREAM_EVENTS = ()  # type: ignore


async def _emit_investigative_event(
    broadcaster: Callable[[str, dict], Awaitable[None]],
    session_id: str,
    event_type: str,
    **kwargs: object,
) -> None:
    """Emit a typed investigative stream event via the WS broadcaster.

    Args:
        broadcaster: Async callable that sends payloads to the WS client.
        session_id: Current agent session identifier.
        event_type: One of REQUIRED_STREAM_EVENTS (phase_change, thread_started, …).
        **kwargs: Additional fields merged into the event payload.
    """
    try:
        payload = {"type": event_type, **kwargs}
        await broadcaster(session_id, payload)
        log.debug("investigative event emitted: %s for session %s", event_type, session_id)
    except Exception as exc:  # pragma: no cover
        log.warning("investigative event emission failed (%s): %s", event_type, exc)


def _normalize_session_graph_token(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").upper()
    return normalized or fallback


_SESSION_GRAPH_LABEL_ALIASES = {
    "ORGANISATION": "ORGANIZATION",
}


_EU_FULL_NAME = "European Union"

_SESSION_GRAPH_NAME_ALIASES = {
    "EU": _EU_FULL_NAME,
    "E U": _EU_FULL_NAME,
    "EUROPEAN UNION": _EU_FULL_NAME,
}


def _canonicalize_session_graph_label(value: str) -> str:
    label = _normalize_session_graph_token(value, "ENTITY")
    return _SESSION_GRAPH_LABEL_ALIASES.get(label, label)


def _canonicalize_session_graph_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n.,;:")
    if not name:
        return ""
    alias_key = re.sub(r"[^A-Za-z0-9]+", " ", name).strip().upper()
    return _SESSION_GRAPH_NAME_ALIASES.get(alias_key, name)


def _session_graph_label_rank(label: str) -> int:
    if label == "ENTITY":
        return 0
    return 1


def _normalize_session_graph_payload(  # noqa: C901  # NOSONAR
    nodes: list[dict],
    edges: list[dict],
) -> tuple[list[dict], list[dict]]:
    normalized_nodes: list[dict] = []
    normalized_edges: list[dict] = []
    canonical_nodes: dict[str, dict] = {}
    alias_to_canonical: dict[str, str] = {}

    for node in nodes:
        node_id = node.get("id", "")
        raw_name = node.get("name") or node_id
        canonical_name = _canonicalize_session_graph_name(
            raw_name
        ) or _canonicalize_session_graph_name(node_id)
        canonical_label = _canonicalize_session_graph_label(
            node.get("type") or node.get("label") or "ENTITY"
        )
        if not canonical_name:
            continue

        alias_to_canonical[str(node_id or "")] = canonical_name
        alias_to_canonical[str(raw_name or "")] = canonical_name
        alias_to_canonical[_canonicalize_session_graph_name(node_id)] = canonical_name
        alias_to_canonical[_canonicalize_session_graph_name(raw_name)] = canonical_name

        existing = canonical_nodes.get(canonical_name)
        if existing is None:
            canonical_nodes[canonical_name] = {
                "id": canonical_name,
                "label": canonical_label,
                "properties": {"name": canonical_name},
            }
            continue

        if _session_graph_label_rank(canonical_label) > _session_graph_label_rank(
            existing["label"]
        ):
            existing["label"] = canonical_label

    normalized_nodes.extend(canonical_nodes.values())

    for edge in edges:
        raw_source = edge.get("source", "")
        raw_target = edge.get("target", "")
        source = (
            alias_to_canonical.get(str(raw_source))
            or alias_to_canonical.get(_canonicalize_session_graph_name(raw_source))
            or _canonicalize_session_graph_name(raw_source)
        )
        target = (
            alias_to_canonical.get(str(raw_target))
            or alias_to_canonical.get(_canonicalize_session_graph_name(raw_target))
            or _canonicalize_session_graph_name(raw_target)
        )
        if not source or not target:
            continue

        properties = {
            key: value for key, value in edge.items() if key not in {"source", "target", "type"}
        }
        normalized_edges.append(
            {
                "source": source,
                "target": target,
                "type": _normalize_session_graph_token(
                    edge.get("type", "RELATED_TO"), "RELATED_TO"
                ),
                "properties": properties,
            }
        )

    deduped_edges: list[dict] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in normalized_edges:
        edge_key = (
            edge["source"],
            edge["target"],
            edge["type"],
            json.dumps(edge["properties"], sort_keys=True),
        )
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        deduped_edges.append(edge)

    return normalized_nodes, deduped_edges


# ── Stopwords ────────────────────────────────────────────────────────────────
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "and",
        "or",
        "for",
        "is",
        "are",
        "was",
        "were",
        "on",
        "at",
        "by",
        "with",
        "this",
        "that",
        "as",
        "be",
        "has",
        "have",
        "its",
        "it",
        "from",
        "not",
        "but",
        "their",
        "they",
        "which",
        "who",
        "will",
        "been",
        "more",
        "about",
        "also",
        "after",
        "before",
        "than",
        "into",
        "through",
        "during",
        "between",
        "against",
        "each",
        "other",
        "those",
        "these",
        "such",
        "when",
        "where",
        "would",
    }
)


def _query_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", (text or "").lower())
        if token not in _STOPWORDS
    }


def _score_query_overlap(query: str, title: str, text: str) -> int:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0
    document_terms = _query_terms(f"{title or ''} {text or ''}")
    return len(query_terms & document_terms)


_SEARCH_WEB_NAV_PHRASES = frozenset(
    {
        "about us",
        "boards",
        "contact us",
        "email us",
        "experts",
        "home",
        "library",
        "members login",
        "newsletter",
        "privacy policy",
        "staff",
        "subscribe",
        "the institute",
    }
)

_SEARCH_WEB_MENU_PREFIX_RE = re.compile(r"^(?:[A-Za-z][A-Za-z&'\- ]{1,30}\s*[/|•·]\s*){3,}")


def _clean_search_web_fragment(text: str) -> str:
    cleaned = re.sub(r"[\t\r\n]+", " ", text or "")
    cleaned = _SEARCH_WEB_MENU_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -•|/:;")


def _search_web_noise_penalty(text: str) -> int:
    lowered = (text or "").lower()
    penalty = sum(1 for phrase in _SEARCH_WEB_NAV_PHRASES if phrase in lowered)

    separator_parts = [
        part.strip() for part in re.split(r"\s*[|/•·»]\s*", text or "") if part.strip()
    ]
    short_parts = sum(1 for part in separator_parts if len(part.split()) <= 3)
    if len(separator_parts) >= 4 and short_parts >= len(separator_parts) - 1:
        penalty += 4

    return penalty


def _iter_search_web_candidates(raw_content: str, content: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        cleaned = _clean_search_web_fragment(text)
        if len(cleaned) < 35:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(cleaned)

    _add(content)

    raw_text = raw_content or ""
    for block in re.split(r"[\r\n]+", raw_text):
        _add(block)
        sentences = re.split(r"(?<=[.!?])\s+", block)
        for index, sentence in enumerate(sentences):
            _add(sentence)
            if index + 1 < len(sentences):
                _add(f"{sentence} {sentences[index + 1]}")

    collapsed = re.sub(r"[\r\n]+", " ", raw_text)
    for sentence in re.split(r"(?<=[.!?])\s+", collapsed):
        _add(sentence)

    return candidates


def _choose_search_web_snippet(
    query: str,
    title: str,
    raw_content: str,
    content: str,
    max_chars: int = 400,
) -> str:
    candidates = _iter_search_web_candidates(raw_content, content)
    if not candidates:
        return _clean_search_web_fragment(title or content or raw_content)[:max_chars]

    def _score(candidate: str) -> tuple[int, int, int, int]:
        return (
            _score_query_overlap(query, "", candidate),
            -_search_web_noise_penalty(candidate),
            1 if re.search(r"[.!?]", candidate) else 0,
            min(len(candidate), max_chars),
        )

    return max(candidates, key=_score)[:max_chars]


_SEARCH_EU_SAFE_SOURCE_IDS = ("eprs_briefings", "ep_resolutions", "ep_debates")


def _normalize_search_eu_sources(sources: str) -> list[str]:
    requested_source_ids = [
        source_id.strip() for source_id in (sources or "").split(",") if source_id.strip()
    ]
    normalized_source_ids = list(
        dict.fromkeys(
            source_id
            for source_id in requested_source_ids
            if source_id in _SEARCH_EU_SAFE_SOURCE_IDS
        )
    )

    # Replace empty or noisy mixes with the validated briefing path.
    # Accept any recognized source; fill with defaults only if empty
    if not normalized_source_ids:
        return list(_SEARCH_EU_SAFE_SOURCE_IDS)
    return normalized_source_ids


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

        raw_terms = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)
        terms = [t for t in raw_terms if t.lower() not in _STOPWORDS and len(t) > 3]
        unique_terms = list(dict.fromkeys(terms))[:15]

        doc_node_id = f"DOC::{doc_id}"
        if doc_node_id not in seen_terms:
            nodes.append(
                {
                    "id": doc_node_id,
                    "type": "Document",
                    "name": doc_title[:60],
                    "doc_id": doc_id,
                    "url": doc.get("url", ""),
                }
            )
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
        title_terms = re.findall(r"\b[A-Z][a-zA-Z]+\b", doc.get("title", ""))
        proper_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        all_terms = title_terms + proper_terms + acronyms
        unique = list(
            dict.fromkeys(t for t in all_terms if t.lower() not in _STOPWORDS and len(t) > 2)
        )[:15]

        for term in unique:
            if term not in seen:
                nodes.append({"id": term, "type": "CONCEPT", "name": term})
                seen.add(term)
                entity_names.append(term)
        for i, a in enumerate(unique):
            for b in unique[i + 1 : i + 4]:
                edges.append({"source": a, "target": b, "type": "CO_OCCURS"})

    lexical = _build_lexical_graph(documents)
    try:
        await broadcaster(
            session_id,
            {
                "type": "graph_update",
                "graph": {"nodes": nodes, "edges": edges},
                "lexical": lexical,
                "partial": True,  # signal: GLiNER2 graph is still running
            },
        )
    except Exception:
        pass

    return entity_names


# ── GLiNER2 + DSPy in-process graphgen ───────────────────────────────────────
async def _run_graphgen(
    documents: list[dict],
    session_id: str,
    broadcaster: Callable[[str, dict], Awaitable[None]],
    persist_graph: Callable[[str, list[dict], list[dict]], Awaitable[None]] | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Run GLiNER2+DSPy graphgen in-process. Returns (nodes, edges, summary_str)."""
    lexical = _build_lexical_graph(documents)

    try:
        docs_input = [
            {
                "id": d.get("id", d.get("url", "")[:60]),
                "title": d.get("title", ""),
                "text": str(d.get("text", ""))[:2000],
                "source": d.get("source", ""),
            }
            for d in documents[:10]
        ]

        def _extract_in_thread() -> dict:
            # Give this thread a fresh event loop so DSPy's internal
            # run_until_complete() doesn't collide with the running asyncio loop.
            import asyncio as _asyncio

            _asyncio.set_event_loop(_asyncio.new_event_loop())
            from services.agent.tools.extraction.in_process import InProcessExtractor

            _extractor = InProcessExtractor()
            return _extractor.extract(docs_input, session_id)

        result = await asyncio.to_thread(_extract_in_thread)
    except Exception as exc:
        log.exception("In-process extraction failed for session=%s", session_id)
        raise RuntimeError(exc) from exc

    nodes = result.get("entities", [])[:300]
    edges = result.get("relations", [])[:600]
    n_nodes, n_edges = len(nodes), len(edges)
    summary = f"Graph built: {n_nodes} entities, {n_edges} relations (GLiNER2 + DSPy)"

    normalized_nodes, normalized_edges = _normalize_session_graph_payload(nodes, edges)

    # Store canonical entity names for LLM reasoning
    entity_names = list(
        dict.fromkeys(
            node.get("properties", {}).get("name", node.get("id", "")) for node in normalized_nodes
        )
    )[:50]
    _session_entities.setdefault(session_id, [])
    existing = set(_session_entities[session_id])
    _session_entities[session_id].extend(e for e in entity_names if e not in existing)

    if persist_graph is not None:
        try:
            await persist_graph(session_id, normalized_nodes, normalized_edges)
        except Exception:
            log.exception("Graph persistence callback failed for %s", session_id)
            raise

    try:
        await broadcaster(
            session_id,
            {
                "type": "graph_update",
                "graph": {"nodes": nodes, "edges": edges},
                "lexical": lexical,
                "partial": False,
            },
        )
    except Exception:
        pass

    log.info("Graph update broadcast: %d nodes, %d edges for %s", n_nodes, n_edges, session_id)
    return nodes, edges, summary


# ── LangGraph agent factory ──────────────────────────────────────────────────
def build_graph(  # noqa: C901  # NOSONAR
    ws_broadcaster: Callable[[str, dict], Awaitable[None]] | None,
    session_id: str,
    openrouter_api_key: str,
    persist_graph: Callable[[str, list[dict], list[dict]], Awaitable[None]] | None = None,
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
            log.debug("search_web  query=%r  raw_results=%d", query, len(results))
            docs_to_store = []
            lines = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = _choose_search_web_snippet(
                    query=query,
                    title=title,
                    raw_content=r.get("raw_content") or "",
                    content=r.get("content") or "",
                )
                date = r.get("published_date", "")
                log.debug("  [%d] url=%r  snippet_len=%d", i, url[:80], len(snippet))
                lines.append(f"{i}. {title}\n   URL: {url}\n   Date: {date}\n   {snippet}")
                docs_to_store.append(
                    {
                        "id": url[:80],
                        "title": title,
                        "url": url,
                        "text": snippet,
                        "date": date,
                        "source": "web",
                    }
                )

            _session_docs.setdefault(session_id, []).extend(docs_to_store)
            log.info("search_web  stored %d docs  session=%s", len(docs_to_store), session_id)

            # Emit evidence_added event for each new document
            for doc in docs_to_store:
                _task = asyncio.create_task(
                    _emit_investigative_event(
                        broadcaster,
                        session_id,
                        "evidence_added",
                        evidence_id=doc.get("id", ""),
                        title=doc.get("title", ""),
                        url=doc.get("url", ""),
                        source="search_web",
                    )
                )

            # Fire partial keyword graph immediately — non-blocking background task
            if docs_to_store:
                _task = asyncio.create_task(
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
        sources: str = "eprs_briefings",
        max_docs: int = 8,
    ) -> str:
        """Search EU official public archives: EP debates, EUR-Lex, Council, EC press releases.

        After results are returned, a partial knowledge graph is automatically
        streamed to the UI — you do NOT need to call graphgen manually yet.

        Args:
            query: Search query string.
            sources: Comma-separated source IDs, normalized server-side to reliable briefing sources.
            max_docs: Maximum documents to return (default 8).
        """
        # Delegate to the search service over HTTP — no direct adapter imports.
        from agent.tools.search.client import SearchServiceClient

        source_ids = _normalize_search_eu_sources(sources)
        log.debug("search_eu  query=%r  sources=%s  max_docs=%d", query, source_ids, max_docs)
        documents: list[dict] = []

        client = SearchServiceClient()
        try:
            raw_docs = await asyncio.wait_for(
                client.fetch_documents(
                    query=query,
                    sources=source_ids[:4],
                    limit=max_docs,
                ),
                timeout=30,
            )
            documents = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "text": str(doc.text or "")[:4000],
                    "url": doc.url,
                    "source": doc.source,
                    "date": doc.date or "",
                }
                for doc in raw_docs
            ]
        except asyncio.TimeoutError:
            log.warning("search_eu timed out for sources %s", source_ids)
        except Exception as exc:
            log.warning("search_eu failed: %s", exc)

        log.debug("search_eu  fetched %d raw docs from sources=%s", len(documents), source_ids)
        if not documents:
            return "No EU documents found for this query."

        _session_docs.setdefault(session_id, []).extend(documents[:max_docs])
        log.info("search_eu  stored %d docs  session=%s", len(documents[:max_docs]), session_id)

        # Fire partial keyword graph immediately — non-blocking
        if documents:
            _task = asyncio.create_task(
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

        log.debug(
            "graphgen  session=%s  docs_accumulated=%d", session_id_arg or session_id, len(docs)
        )
        try:
            _, _, summary = await _run_graphgen(
                docs,
                session_id_arg or session_id,
                broadcaster,
                persist_graph,
            )
        except Exception:
            log.exception(
                "graphgen extraction failed for session=%s",
                session_id_arg or session_id,
            )
            raise

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

    # ── Tool 5: search_archives ──────────────────────────────────────────────
    @tool
    async def search_archives(
        query: str,
        archives: str,
        date_from: str = "",
        date_to: str = "",
        limit_per_archive: int = 5,
    ) -> str:
        """Search specific high-quality public archives for authoritative primary sources.

        Use this tool when you need targeted depth from specific official archives —
        legislation text, parliamentary debates, court decisions, central bank reports.
        Complements search_eu (which uses EPRS/EP sources) and search_web (open web).
        Results are cosine-reranked against the query and cached in the knowledge graph.

        ARCHIVE IDs — pick only the most relevant ones for the query:
          eprs_briefings       EU policy briefs & impact assessments (best for policy overview)
          eurlex               EU legislation: regulations, directives, treaties (primary law text)
          ep_debates           EP plenary debate transcripts (MEP positions, political dynamics)
          ep_resolutions       EP adopted resolutions (Parliament's official position)
          council_conclusions  Council of EU conclusions & press releases (member state positions)
          ec_press             European Commission press releases (executive announcements)
          ecb_publications     ECB monetary policy, financial stability reports
          uk_legislation       UK Acts of Parliament & Statutory Instruments
          wikipedia_eu         Wikipedia background context (secondary source only)

        Args:
            query: Natural language search query.
            archives: Comma-separated archive IDs from the list above.
            date_from: Optional ISO date filter YYYY-MM-DD.
            date_to: Optional ISO date filter YYYY-MM-DD.
            limit_per_archive: Documents to fetch per archive (default 5, max 10).
        """
        _SERVICES_DIR = Path(__file__).resolve().parents[1]
        if str(_SERVICES_DIR) not in sys.path:
            sys.path.insert(0, str(_SERVICES_DIR))

        from search.archive_engine import ArchiveEngine, ArchiveQuery

        archive_ids = [a.strip() for a in archives.split(",") if a.strip()]
        if not archive_ids:
            return "No archives specified. Provide comma-separated archive IDs."

        log.debug(
            "search_archives  query=%r  archives=%s  limit_per_archive=%d",
            query,
            archive_ids,
            limit_per_archive,
        )
        aq = ArchiveQuery(
            query=query,
            archives=archive_ids,
            date_from=date_from or None,
            date_to=date_to or None,
            limit_per_archive=min(limit_per_archive, 10),
            top_k=8,
        )

        engine = ArchiveEngine()
        results = await engine.search(aq)
        formatted = engine.format_for_agent(results)

        # Accumulate docs into session store for graphgen
        docs_for_session = [
            {
                "id": r.doc_id,
                "title": r.title,
                "text": r.excerpt,
                "url": r.url,
                "source": r.source,
                "date": r.date,
            }
            for r in results
        ]
        _session_docs.setdefault(session_id, []).extend(docs_for_session)

        # Emit keyword graph (non-blocking)
        if docs_for_session:
            _task = asyncio.create_task(
                _emit_keyword_graph(
                    _session_docs.get(session_id, []),
                    session_id,
                    broadcaster,
                )
            )

        cached_count = sum(1 for r in results if r.from_cache)
        log.info(
            "search_archives  %d results (%d cached)  archives=%s  session=%s",
            len(results),
            cached_count,
            archive_ids,
            session_id,
        )
        for _r in results:
            log.debug(
                "  result  source=%s  doc_id=%r  score=%.3f  cached=%s",
                _r.source,
                _r.doc_id[:60],
                getattr(_r, "score", 0.0),
                _r.from_cache,
            )
        header = (
            f"[search_archives] {len(results)} results from {archive_ids} "
            f"({cached_count} from cache)\n\n"
        )
        return header + formatted

    # ── Tool 4: hybridrag ────────────────────────────────────────────────────
    @tool
    async def hybridrag(query: str, session_id_arg: str, k: int = 10) -> str:
        """Query the hybrid knowledge graph for entity relationships, evidence, and patterns.

        Performs entity-first graph retrieval (Neo4j MENTIONS traversal) fused with
        pgvector semantic search via RRF. Returns an XML <retrieval_context> document
        with graph-matched entities, semantic chunks, and ranked fused results.

        Args:
            query: The research query (e.g. 'who funds Russian disinformation campaigns?').
            session_id_arg: Session ID for scoped chunk retrieval.
            k: Number of fused results to return (default 10).
        """
        sid = session_id_arg or session_id
        import os as _os

        hybridrag_url = _os.environ.get("HYBRIDRAG_URL", "http://localhost:8020")
        log.debug(
            "hybridrag  query=%r  session=%s  k=%d  url=%s", query[:80], sid, k, hybridrag_url
        )
        try:
            from services.agent.tools.graph.client import HybridRAGClient

            client = HybridRAGClient(hybridrag_url, timeout=25.0)
            xml_result = await client.retrieve(
                query,
                session_id=sid,
                k_entity=k,
                k_semantic=k * 2,
                k_final=k,
            )
            log.info("hybridrag  result_len=%d chars  session=%s", len(xml_result), sid)
            return xml_result
        except Exception as exc:
            entities = _session_entities.get(sid, [])
            if entities:
                return (
                    f"HybridRAG service unavailable. Session entities so far: "
                    f"{', '.join(entities[:20])}"
                )
            return f"HybridRAG unavailable: {exc}"

    # ── System prompt ────────────────────────────────────────────────────────
    from datetime import date as _date

    today = _date.today().isoformat()

    SYSTEM_PROMPT = f"""You are an EU public records investigator for the European Parliamentary Research Service.
Today's date: {today}
Session ID: {session_id}

RESEARCH PROTOCOL — follow this EXACT sequence:

STEP 1 (one LLM response): Call search_eu AND search_web simultaneously.
  → search_eu: Use sources="eprs_briefings,ep_resolutions,ep_debates" — these are our primary authoritative EU sources.
    EPRS briefings = EP Research Service analysis. EP resolutions = adopted Parliament texts. EP debates = plenary speeches.
  → search_web: Use for context, news coverage, and any sources not in EU official archives.
  → A partial keyword graph is automatically streamed to the user as you search.

STEP 1b (optional, same response as STEP 1): Call search_archives if the query benefits from targeted depth.
  → Use search_archives when you need: specific legislation text (eurlex), ECB/monetary data (ecb_publications),
    UK law (uk_legislation), Council positions (council_conclusions), or Commission announcements (ec_press).
  → Pick only the 1-2 most relevant archive IDs — do not fan out to all archives.
  → search_archives complements search_eu and search_web — it does not replace them.
  → Results are cosine-reranked and cached in the knowledge graph automatically.

STEP 2 (next LLM response): Call graphgen(session_id_arg="{session_id}")
  → This runs GLiNER2 + DSPy to extract real named entities and typed relations.
  → The result includes the key entities found — use them in your report.
  → CRITICAL: Call graphgen in a SEPARATE response AFTER receiving search results.

STEP 3 (required): After graphgen returns, call hybridrag to retrieve evidence from the knowledge graph and pgvector store.
  → hybridrag(query=<your research question>, session_id_arg="{session_id}", k=10)
  → This is MANDATORY — it retrieves entity connectivity and semantic evidence accumulated during the session.
  → Do NOT skip this step even if graphgen returned few entities.

STEP 4: Write your final EPRS-style investigative report.

RULES:
- ALWAYS call search_eu with sources="eprs_briefings,ep_resolutions,ep_debates" — never omit this
- Do NOT call graphgen in the same response as search tools (they run in parallel, docs won't be ready)
- ALWAYS call hybridrag after graphgen and before the final report — never skip it
- Do NOT loop or re-search without a clear reason
- The report MUST include a 'Key Graph Entities' section listing the entities from graphgen
- Cite EU sources inline with [🏛️ Source] and web sources with [🌐 Source]
- EU official sources (EPRS, EP resolutions, debates) take priority over web sources

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
    tools_list = [search_web, search_eu, search_archives, graphgen, hybridrag]

    llm_base = ChatOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-chat",
        temperature=0.1,
        streaming=True,
    )
    llm_with_tools = llm_base.bind_tools(tools_list)
    llm_force_tools = llm_base.bind_tools(tools_list, tool_choice="required")

    tool_node = ToolNode(tools_list)

    async def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        is_first_call = not any(getattr(m, "type", None) == "tool" for m in messages)
        if is_first_call:
            first_human = next(
                (m.content for m in messages if getattr(m, "type", None) == "human"), ""
            )
            await _emit_investigative_event(broadcaster, session_id, "phase_change", phase="scope")
            await _emit_investigative_event(
                broadcaster,
                session_id,
                "thread_started",
                thread_id=session_id,
                query=first_human,
            )
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        tool_msgs = [m for m in messages if getattr(m, "type", None) == "tool"]
        called_tool_names = {getattr(m, "name", None) for m in tool_msgs}
        # Force tool calls until the mandatory pipeline (graphgen → hybridrag) is complete.
        # Safety cap at 12 tool rounds to prevent infinite loops.
        required_done = called_tool_names >= {"graphgen", "hybridrag"}
        force = not required_done and len(tool_msgs) < 12

        # If the LLM just produced text without tool calls but required steps
        # aren't done yet, inject a protocol reminder before the next call so
        # the model knows exactly which tool to call.
        last_msg = messages[-1] if messages else None
        last_is_ai_no_tools = (
            last_msg is not None
            and getattr(last_msg, "type", None) == "ai"
            and not getattr(last_msg, "tool_calls", None)
        )
        if force and last_is_ai_no_tools and tool_msgs:
            missing = sorted({"graphgen", "hybridrag"} - called_tool_names)
            messages = list(messages) + [
                HumanMessage(
                    content=(
                        f"PROTOCOL REMINDER: Required steps not yet completed. "
                        f"You MUST call the following tool(s) now before writing the report: "
                        f"{', '.join(missing)}. Do NOT write the report yet."
                    )
                )
            ]

        active_llm = llm_force_tools if force else llm_with_tools
        response = await active_llm.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state):  # no type hints — Python 3.13 get_type_hints compat
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        # Required pipeline (graphgen → hybridrag) not yet complete → loop back
        # to agent so the injected reminder + force-tools can take effect.
        all_msgs = state["messages"]
        tool_msgs_s = [m for m in all_msgs if getattr(m, "type", None) == "tool"]
        if len(tool_msgs_s) < 12:
            called = {getattr(m, "name", None) for m in tool_msgs_s}
            if not (called >= {"graphgen", "hybridrag"}):
                return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "agent": "agent", END: END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
