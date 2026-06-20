"""graphknows dependency-injection container.

``GraphKnowsContainer`` is an async context manager that owns the full store
lifecycle (ArcadeDB STM, LTM, and vector) and surfaces all capabilities as
framework-agnostic ``GraphKnowsTool`` objects.

Typical usage in any agentic project::

    from graphknows import GraphKnows

    async with GraphKnows.from_env() as gk:
        # Framework-agnostic tool list
        tools = gk.tools          # list[GraphKnowsTool]

        # Direct programmatic access
        context = await gk.retriever.retrieve("query", session_id="s1")

No agent framework is imported here.  Callers wrap ``GraphKnowsTool.fn`` for
their stack in a single line — see ``graphknows/tools.py`` for examples.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from graphknows.config import GraphKnowsSettings
from graphknows.config.builders import (
    build_arcadedb_ltm_store,
    build_arcadedb_stm_store,
    build_arcadedb_vector_store,
)
from graphknows.mcp.tool import GraphKnowsTool
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.retriever import DETRetriever
from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore
from graphknows.memory.stores.embedder import embed

log = logging.getLogger("graphknows.container")


class GraphKnowsContainer:
    """Async context manager that owns the full graphknows store lifecycle.

    Attributes:
        settings: Resolved ``GraphKnowsSettings`` instance.
    """

    def __init__(self, settings: GraphKnowsSettings) -> None:
        self.settings = settings
        self._vector: ArcadeDBVectorStore | None = None
        self._ltm: ArcadeDBLTMStore | None = None
        self._stm: ArcadeDBSTMStore | None = None
        self._retriever: BaseGraphRetriever | None = None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._background_tasks: set[asyncio.Task] = set()

    # Sentinel message for guards that detect use outside the async context manager
    _ERR_NOT_ENTERED = "GraphKnowsContainer not entered — use 'async with'"
    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GraphKnowsContainer:
        """Build a container from environment variables / .env file."""
        return cls(GraphKnowsSettings())

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> GraphKnowsContainer:
        cfg = self.settings
        self._stm = build_arcadedb_stm_store(cfg)
        self._ltm = build_arcadedb_ltm_store(cfg)
        self._vector = build_arcadedb_vector_store(cfg)

        await asyncio.gather(
            self._stm._c.connect(),
            self._ltm._c.connect(),
            self._vector._c.connect(),
        )
        await asyncio.gather(
            self._stm.ensure_schema(),
            self._ltm.ensure_schema(),
            self._vector.ensure_schema(),
        )
        if cfg.retrieval_mode == "agentic":
            from graphknows.memory.query.agentic import AgenticRetriever

            self._retriever = AgenticRetriever(
                settings=cfg,
                pg=self._vector,
                neo=self._ltm,
                stm=self._stm,
            )
        else:
            self._retriever = DETRetriever(
                pg=self._vector,
                neo=self._ltm,
                mg=self._stm,
                embed_model=cfg.embed_model,
            )
        log.info("GraphKnowsContainer ready (env=%s)", cfg.graphknows_env)
        return self

    async def __aexit__(self, *_: object) -> None:
        for store in (self._stm, self._ltm, self._vector):
            try:
                if store is not None:
                    await store._c.close()
            except Exception:
                log.debug("Cleanup error ignored", exc_info=True)
        self._vector = None
        self._ltm = None
        self._stm = None
        self._retriever = None
        log.info("GraphKnowsContainer closed")

    # ------------------------------------------------------------------
    # Store accessors (fail-fast if used outside context manager)
    # ------------------------------------------------------------------

    @property
    def retriever(self) -> BaseGraphRetriever:
        """Return the active retriever; raises if container is not entered."""
        if self._retriever is None:
            raise RuntimeError(self._ERR_NOT_ENTERED)
        return self._retriever

    @property
    def vector(self) -> ArcadeDBVectorStore:
        """Return the vector store; raises if container is not entered."""
        if self._vector is None:
            raise RuntimeError(self._ERR_NOT_ENTERED)
        return self._vector

    @property
    def ltm(self) -> ArcadeDBLTMStore:
        """Return the LTM graph store; raises if container is not entered."""
        if self._ltm is None:
            raise RuntimeError(self._ERR_NOT_ENTERED)
        return self._ltm

    @property
    def stm(self) -> ArcadeDBSTMStore:
        """Return the STM store; raises if container is not entered."""
        if self._stm is None:
            raise RuntimeError(self._ERR_NOT_ENTERED)
        return self._stm

    # ------------------------------------------------------------------
    # Framework-agnostic tool list
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[GraphKnowsTool]:
        """All graphknows capabilities as framework-agnostic ``GraphKnowsTool`` objects.

        Tools are instantiated lazily on first access and bound to this
        container's live stores.  The container must be entered before
        any tool is *called* (though building the list is safe beforehand).
        """
        return _build_tools(self)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


async def _embed_one(text: str, model: str) -> list[float]:
    """Embed a single string and return a plain float list."""
    arr = await asyncio.to_thread(embed, [text], model)
    return arr[0].tolist()


# ---------------------------------------------------------------------------
# Tool implementations — each receives the container as its first argument.
# _build_tools() binds `c` via functools.partial so callers see only the
# domain parameters.
# ---------------------------------------------------------------------------


async def _tool_retrieve(
    c: GraphKnowsContainer,
    query: str,
    session_id: str = "default",
    top_k: int = 10,
) -> str:
    ctx = await c.retriever.retrieve(query, session_id, top_k=top_k)
    return ctx.to_xml()


async def _tool_entity_search(
    c: GraphKnowsContainer,
    query: str,
    k: int = 10,
    entity_types: list[str] | None = None,
) -> str:
    emb = await _embed_one(query, c.settings.embed_model)
    results = await c.vector.search_entities(emb, k=k, entity_types=entity_types or None)
    return json.dumps(results, ensure_ascii=False)


async def _tool_entity_neighbors(
    c: GraphKnowsContainer,
    entity_name: str,
    depth: int = 1,
    relationship_types: list[str] | None = None,
) -> str:
    result = await c.ltm.get_entity_neighbors(
        entity_name, depth=depth, relationship_types=relationship_types
    )
    return json.dumps(result, ensure_ascii=False)


async def _tool_get_chunks(c: GraphKnowsContainer, chunk_ids: list[str]) -> str:
    text_map = await c.vector.get_chunks_by_ids(chunk_ids)
    entity_results = await asyncio.gather(
        *[c.ltm.get_chunk_entities(cid) for cid in chunk_ids],
        return_exceptions=True,
    )
    enriched: dict[str, Any] = {}
    for cid, ent_result in zip(chunk_ids, entity_results, strict=False):
        base = text_map.get(cid, {})
        enriched[cid] = {
            "doc_id": base.get("doc_id", ""),
            "text": base.get("text", ""),
            "metadata": base.get("metadata", {}),
            "entities": ent_result if not isinstance(ent_result, Exception) else [],
        }
    return json.dumps(enriched, ensure_ascii=False)


async def _tool_session_memory(
    c: GraphKnowsContainer,
    content: str,
    title: str = "",
    session_id: str = "default",
) -> str:
    full_text = f"{title}\n\n{content}" if title else content
    chunk_id = f"memo:{uuid.uuid4().hex}"
    emb = await _embed_one(full_text, c.settings.embed_model)
    await c.vector.upsert_chunks(
        session_id=session_id,
        chunks=[
            {
                "id": chunk_id,
                "doc_id": f"memo:{session_id}",
                "text": full_text,
                "embedding": emb,
                "metadata": {"title": title, "source": "session_memory"},
            }
        ],
    )
    return json.dumps({"chunk_id": chunk_id, "session_id": session_id, "stored": True})


async def _tool_ingest(
    c: GraphKnowsContainer,
    text: str,
    title: str = "",
    config_override: dict[str, Any] | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    queued_at = datetime.now(UTC).isoformat()
    c._jobs[run_id] = {"status": "queued", "queued_at": queued_at}

    async def _run_workflow() -> None:
        c._jobs[run_id]["status"] = "running"
        c._jobs[run_id]["started_at"] = datetime.now(UTC).isoformat()
        try:
            from graphknows.memory.extractor import GLiNER2EntityExtractor
            from graphknows.memory.workflows.corpus_ingest import (
                CorpusIngestRequest,
                CorpusIngestWorkflow,
            )

            extractor = GLiNER2EntityExtractor()
            workflow = CorpusIngestWorkflow(
                mg_store=c._stm,
                pg_store=c._vector,
                neo_store=c._ltm,
                extractor=extractor,
                embed_fn=None,
            )
            request = CorpusIngestRequest(
                documents=[{"text": text, "title": title}],
            )
            result = await workflow.run(request)
            stats = {
                "session_id": result.session_id,
                "documents_processed": result.documents_processed,
                "entities_extracted": result.entities_extracted,
                "chunks_written": result.chunks_written,
                "nodes_written": result.nodes_written,
                "edges_written": result.edges_written,
                "elapsed_s": result.elapsed_s,
                "errors": result.errors,
            }
            c._jobs[run_id].update(
                status="completed",
                stats=stats,
                finished_at=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:
            log.exception("Ingest job %s failed", run_id)
            c._jobs[run_id].update(
                status="failed",
                error=str(exc),
                finished_at=datetime.now(UTC).isoformat(),
            )

    bg_task = asyncio.create_task(_run_workflow())
    c._background_tasks.add(bg_task)
    bg_task.add_done_callback(c._background_tasks.discard)
    return json.dumps({"run_id": run_id, "status": "queued", "queued_at": queued_at})


async def _tool_poll_job(c: GraphKnowsContainer, run_id: str) -> str:
    job = c._jobs.get(run_id)
    if job is None:
        return json.dumps({"run_id": run_id, "status": "not_found"})
    return json.dumps({"run_id": run_id, **job})


async def _tool_get_session_graph(c: GraphKnowsContainer, session_id: str = "default") -> str:
    result = await c.stm.get_session_graph(session_id)
    return json.dumps(result, ensure_ascii=False)


async def _tool_get_corpus_stats(c: GraphKnowsContainer) -> str:
    neo_counts = await c.ltm.count_nodes()
    stm_rows = await c.vector._c.query(
        "stm", "MATCH (n:CHUNK) RETURN count(n) AS c", language="cypher"
    )
    ltm_rows = await c.vector._c.query(
        "ltm", "MATCH (n:CHUNK) RETURN count(n) AS c", language="cypher"
    )
    stm_turns = await c.vector._c.query(
        "stm", "MATCH (n:TURN) RETURN count(n) AS c", language="cypher"
    )
    ltm_turns = await c.vector._c.query(
        "ltm", "MATCH (n:TURN) RETURN count(n) AS c", language="cypher"
    )
    vector_counts = {
        "stm_chunks": int(stm_rows[0].get("c") or 0) if stm_rows else 0,
        "ltm_chunks": int(ltm_rows[0].get("c") or 0) if ltm_rows else 0,
        "stm_turns": int(stm_turns[0].get("c") or 0) if stm_turns else 0,
        "ltm_turns": int(ltm_turns[0].get("c") or 0) if ltm_turns else 0,
    }
    return json.dumps({"graph": neo_counts, "vectors": vector_counts})


async def _tool_community_summary(c: GraphKnowsContainer, entity_name: str) -> str:
    topic = await c.ltm.get_entity_community(entity_name)
    if topic is None:
        return f"No community found for entity: {entity_name}"
    return f"{topic['label']}\n\n{topic['summary']}"


async def _tool_list_segments(c: GraphKnowsContainer, limit: int = 20) -> str:
    segments = await c.ltm.list_segments(limit=limit)
    if not segments:
        return "No SEGMENT nodes found in the corpus."
    lines = ["SEGMENT nodes (newest first):", ""]
    for seg in segments:
        lines.append(
            f"  {seg.get('segment_id', 'unknown')}  |  "
            f"{seg.get('ingested_at', 'N/A')}  |  "
            f"taxonomy={seg.get('taxonomy_name', 'N/A')}  |  "
            f"docs={seg.get('source_count', '?')}"
        )
    return "\n".join(lines)


async def _tool_segment_info(c: GraphKnowsContainer, segment_id: str) -> str:
    info = await c.ltm.get_segment_info(segment_id)
    if info is None:
        return f"No SEGMENT found with id: {segment_id}"
    lines = [f"SEGMENT: {segment_id}", ""]
    for k, v in info.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


async def _tool_cypher_query(
    c: GraphKnowsContainer,
    template: str,
    params: dict[str, Any] | None = None,
) -> str:
    result = await c.ltm.run_read_query(template, params or {})
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tool builder — assembles the list[GraphKnowsTool] bound to a container
# ---------------------------------------------------------------------------


def _build_tools(c: GraphKnowsContainer) -> list[GraphKnowsTool]:
    """Instantiate all 13 graphknows tools bound to container ``c``."""

    def _bind(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        return functools.partial(fn, c)

    return [
        GraphKnowsTool(
            name="retrieve",
            description="Search the knowledge graph. Returns XML <retrieval_context>.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language research question.",
                    },
                    "session_id": {"type": "string", "default": "default"},
                    "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
            fn=_bind(_tool_retrieve),
            tags=["retrieval"],
        ),
        GraphKnowsTool(
            name="entity_search",
            description="Discover named entities in the corpus by semantic similarity. Returns JSON array of {name, label, score}.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 10},
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Optional label filter, e.g. ["ORG", "PERSON"].',
                    },
                },
                "required": ["query"],
            },
            fn=_bind(_tool_entity_search),
            tags=["retrieval", "graph"],
        ),
        GraphKnowsTool(
            name="entity_neighbors",
            description="Traverse the knowledge graph from a named entity. Returns neighbors up to depth hops away.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "depth": {"type": "integer", "default": 1, "minimum": 1, "maximum": 3},
                    "relationship_types": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity_name"],
            },
            fn=_bind(_tool_entity_neighbors),
            tags=["graph"],
        ),
        GraphKnowsTool(
            name="get_chunks",
            description="Fetch full chunk text and provenance for a list of chunk IDs from retrieve results.",
            parameters={
                "type": "object",
                "properties": {
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["chunk_ids"],
            },
            fn=_bind(_tool_get_chunks),
            tags=["retrieval"],
        ),
        GraphKnowsTool(
            name="session_memory",
            description="Store a text note in the session-scoped vector store, making it retrievable in future calls.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "title": {"type": "string", "default": ""},
                    "session_id": {"type": "string", "default": "default"},
                },
                "required": ["content"],
            },
            fn=_bind(_tool_session_memory),
            tags=["memory"],
        ),
        GraphKnowsTool(
            name="ingest",
            description="Add a document to the persistent knowledge graph corpus (non-blocking). Returns run_id; use poll_job to check completion.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "title": {"type": "string", "default": ""},
                    "config_override": {"type": "object"},
                },
                "required": ["text"],
            },
            fn=_bind(_tool_ingest),
            tags=["ingestion"],
        ),
        GraphKnowsTool(
            name="poll_job",
            description="Check the status of an ingest job. Returns status: queued | running | completed | failed.",
            parameters={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
            },
            fn=_bind(_tool_poll_job),
            tags=["ingestion"],
        ),
        GraphKnowsTool(
            name="get_session_graph",
            description="Return the entity graph for a session from ArcadeDB STM. Shows entities and RELATES edges.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "default": "default"},
                },
                "required": [],
            },
            fn=_bind(_tool_get_session_graph),
            tags=["graph", "memory"],
        ),
        GraphKnowsTool(
            name="get_corpus_stats",
            description="Return combined node and row counts from ArcadeDB LTM and vector store.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            fn=_bind(_tool_get_corpus_stats),
            tags=["admin"],
        ),
        GraphKnowsTool(
            name="community_summary",
            description="Return the Leiden community topic summary for a named entity.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                },
                "required": ["entity_name"],
            },
            fn=_bind(_tool_community_summary),
            tags=["graph"],
        ),
        GraphKnowsTool(
            name="list_segments",
            description="List temporal SEGMENT provenance nodes (pipeline-run containers), newest first.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                },
                "required": [],
            },
            fn=_bind(_tool_list_segments),
            tags=["admin", "provenance"],
        ),
        GraphKnowsTool(
            name="segment_info",
            description="Return provenance and statistics for a single SEGMENT pipeline-run node.",
            parameters={
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
                },
                "required": ["segment_id"],
            },
            fn=_bind(_tool_segment_info),
            tags=["admin", "provenance"],
        ),
        GraphKnowsTool(
            name="cypher_query",
            description="Run a parameterized read-only Cypher query against ArcadeDB LTM. Write keywords are rejected.",
            parameters={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "description": "Parameterized Cypher template using $param syntax.",
                    },
                    "params": {"type": "object", "description": "Query parameters dict."},
                },
                "required": ["template"],
            },
            fn=_bind(_tool_cypher_query),
            tags=["graph", "admin"],
        ),
    ]
