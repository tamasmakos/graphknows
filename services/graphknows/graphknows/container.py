"""graphknows dependency-injection container.

``GraphKnowsContainer`` is an async context manager that owns the full store
lifecycle (Neo4j, pgvector, Memgraph) and surfaces all capabilities as
framework-agnostic ``GraphKnowsTool`` objects.

Typical usage in any agentic project::

    from graphknows import GraphKnows

    async with GraphKnows.from_env() as gk:
        # Option A — framework-agnostic tool list
        tools = gk.tools          # list[GraphKnowsTool]

        # Option B — MCP server (for out-of-process / MCP-native agents)
        mcp_srv = gk.as_mcp_server()

        # Option C — direct programmatic access
        context = await gk.retriever.retrieve("query", session_id="s1")

No agent framework is imported here.  Callers wrap ``GraphKnowsTool.fn`` for
their stack in a single line — see ``graphknows/tools.py`` for examples.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from graphknows.config import GraphKnowsSettings
from graphknows.models.tools import GraphKnowsTool
from graphknows.retrieval.retriever import HybridRetriever
from graphknows.retrieval.stores.embedder import embed
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.neo4j import Neo4jStore
from graphknows.retrieval.stores.pgvector import PgVectorStore

logger = logging.getLogger("graphknows.container")


class GraphKnowsContainer:
    """Async context manager that owns the full graphknows store lifecycle.

    Attributes:
        settings: Resolved ``GraphKnowsSettings`` instance.
    """

    def __init__(self, settings: GraphKnowsSettings) -> None:
        self.settings = settings
        self._pg: PgVectorStore | None = None
        self._neo: Neo4jStore | None = None
        self._memgraph: MemgraphStore | None = None
        self._retriever: HybridRetriever | None = None
        self._jobs: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> GraphKnowsContainer:
        """Build a container from environment variables / .env file."""
        return cls(GraphKnowsSettings())

    @classmethod
    def from_settings(cls, settings: GraphKnowsSettings) -> GraphKnowsContainer:
        """Build a container from a pre-constructed settings object."""
        return cls(settings)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> GraphKnowsContainer:
        cfg = self.settings
        self._neo = Neo4jStore(
            uri=cfg.neo4j_uri,
            user=cfg.neo4j_user,
            password=cfg.neo4j_password,
            database=cfg.neo4j_database,
        )
        self._pg = PgVectorStore(dsn=cfg.pgvector_dsn)
        self._memgraph = MemgraphStore(uri=cfg.memgraph_uri)

        await self._pg.connect()
        await asyncio.gather(
            self._pg.ensure_schema(),
            self._neo.ensure_indexes(),
        )
        self._retriever = HybridRetriever(
            pg=self._pg,
            neo=self._neo,
            embed_model=cfg.embed_model,
        )
        logger.info("GraphKnowsContainer ready (env=%s)", cfg.graphknows_env)
        return self

    async def __aexit__(self, *_: object) -> None:
        try:
            if self._pg is not None:
                await self._pg.close()
        except Exception:
            pass
        try:
            if self._neo is not None:
                self._neo.close()
        except Exception:
            pass
        self._pg = None
        self._neo = None
        self._memgraph = None
        self._retriever = None
        logger.info("GraphKnowsContainer closed")

    # ------------------------------------------------------------------
    # Store accessors (fail-fast if used outside context manager)
    # ------------------------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        """Return the hybrid retriever; raises if container is not entered."""
        if self._retriever is None:
            raise RuntimeError("GraphKnowsContainer not entered — use 'async with'")
        return self._retriever

    @property
    def pg(self) -> PgVectorStore:
        """Return the PgVector store; raises if container is not entered."""
        if self._pg is None:
            raise RuntimeError("GraphKnowsContainer not entered — use 'async with'")
        return self._pg

    @property
    def neo(self) -> Neo4jStore:
        """Return the Neo4j store; raises if container is not entered."""
        if self._neo is None:
            raise RuntimeError("GraphKnowsContainer not entered — use 'async with'")
        return self._neo

    @property
    def memgraph(self) -> MemgraphStore:
        """Return the Memgraph store; raises if container is not entered."""
        if self._memgraph is None:
            raise RuntimeError("GraphKnowsContainer not entered — use 'async with'")
        return self._memgraph

    # ------------------------------------------------------------------
    # MCP server (out-of-process / MCP-native agents)
    # ------------------------------------------------------------------

    def as_mcp_server(self) -> Any:
        """Return the graphknows FastMCP server with this container wired in.

        The MCP server already exposes all 13 tools; this method injects the
        container's live store instances into it so MCP calls use the same
        connections as the rest of the container.

        Returns:
            A ``FastMCP`` instance ready to be served or mounted.
        """
        from graphknows.mcp.server import mcp, set_dependencies

        set_dependencies(
            self._retriever,
            self._pg,
            self._neo,
            self._memgraph,
            {"embed_model": self.settings.embed_model},
        )
        return mcp

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

    # ------------------------------------------------------------------
    # Internal: background ingest helper shared by the ingest tool
    # ------------------------------------------------------------------

    async def _run_ingest_task(
        self,
        run_id: str,
        text: str,
        title: str,
        config_override: dict[str, Any],
    ) -> None:
        self._jobs[run_id]["status"] = "running"
        self._jobs[run_id]["started_at"] = datetime.now(UTC).isoformat()
        try:
            from graphknows.graphgen import run_ingestion

            stats = await run_ingestion(
                documents=[{"text": text, "title": title}],
                config_override=config_override or None,
            )
            self._jobs[run_id].update(
                status="completed",
                stats=stats,
                finished_at=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:
            logger.exception("Ingest job %s failed: %s", run_id, exc)
            self._jobs[run_id].update(
                status="failed",
                error=str(exc),
                finished_at=datetime.now(UTC).isoformat(),
            )


# ---------------------------------------------------------------------------
# Tool builder — constructs the list[GraphKnowsTool] bound to a container
# ---------------------------------------------------------------------------


def _build_tools(c: GraphKnowsContainer) -> list[GraphKnowsTool]:  # noqa: C901
    """Instantiate all 13 graphknows tools bound to container ``c``."""

    # ---- retrieve -------------------------------------------------------
    async def _retrieve(
        query: str,
        session_id: str = "default",
        search_type: str = "HYBRID",
        k_entity: int = 10,
        k_semantic: int = 20,
        k_final: int = 10,
    ) -> str:
        ret = c.retriever
        mode = search_type.upper()
        if mode == "SEMANTIC":
            ctx = await ret.retrieve_semantic(query, session_id, k=k_semantic)
        elif mode == "GRAPH_TRAVERSAL":
            ctx = await ret.retrieve_graph_traversal(
                query, session_id, k_entity=k_entity, depth=2, k_final=k_final
            )
        elif mode == "SUMMARIES":
            ctx = await ret.retrieve_summaries(query, session_id, k_entity=k_entity)
        elif mode == "FEELING_LUCKY":
            ctx = await ret.retrieve_feeling_lucky(
                query, session_id, k_entity=k_entity, k_semantic=k_semantic, k_final=k_final
            )
        else:
            ctx = await ret.retrieve(
                query, session_id, k_entity=k_entity, k_semantic=k_semantic, k_final=k_final
            )
        return ctx.to_xml()

    # ---- entity_search --------------------------------------------------
    async def _entity_search(
        query: str,
        k: int = 10,
        entity_types: list[str] | None = None,
    ) -> str:
        emb_array = await asyncio.to_thread(embed, [query], c.settings.embed_model)
        emb = emb_array[0].tolist()
        results = await c.pg.search_entities(emb, k=k, entity_types=entity_types or None)
        return json.dumps(results, ensure_ascii=False)

    # ---- entity_neighbors -----------------------------------------------
    async def _entity_neighbors(
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> str:
        result = await c.neo.get_entity_neighbors(
            entity_name, depth=depth, relationship_types=relationship_types
        )
        return json.dumps(result, ensure_ascii=False)

    # ---- get_chunks -----------------------------------------------------
    async def _get_chunks(chunk_ids: list[str]) -> str:
        text_map = await c.pg.get_chunks_by_ids(chunk_ids)
        entity_results = await asyncio.gather(
            *[c.neo.get_chunk_entities(cid) for cid in chunk_ids],
            return_exceptions=True,
        )
        enriched: dict[str, Any] = {}
        for cid, ent_result in zip(chunk_ids, entity_results):
            base = text_map.get(cid, {})
            enriched[cid] = {
                "doc_id": base.get("doc_id", ""),
                "text": base.get("text", ""),
                "metadata": base.get("metadata", {}),
                "entities": ent_result if not isinstance(ent_result, Exception) else [],
            }
        return json.dumps(enriched, ensure_ascii=False)

    # ---- session_memory -------------------------------------------------
    async def _session_memory(
        content: str,
        title: str = "",
        session_id: str = "default",
    ) -> str:
        full_text = f"{title}\n\n{content}" if title else content
        chunk_id = f"memo:{uuid.uuid4().hex}"
        emb_array = await asyncio.to_thread(embed, [full_text], c.settings.embed_model)
        emb = emb_array[0].tolist()
        await c.pg.upsert_chunks(
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

    # ---- ingest ---------------------------------------------------------
    async def _ingest(
        text: str,
        title: str = "",
        config_override: dict[str, Any] | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        queued_at = datetime.now(UTC).isoformat()
        c._jobs[run_id] = {"status": "queued", "queued_at": queued_at}  # noqa: SLF001
        asyncio.create_task(c._run_ingest_task(run_id, text, title, config_override or {}))  # noqa: SLF001
        return json.dumps({"run_id": run_id, "status": "queued", "queued_at": queued_at})

    # ---- poll_job -------------------------------------------------------
    async def _poll_job(run_id: str) -> str:
        job = c._jobs.get(run_id)  # noqa: SLF001
        if job is None:
            return json.dumps({"run_id": run_id, "status": "not_found"})
        return json.dumps({"run_id": run_id, **job})

    # ---- get_session_graph ----------------------------------------------
    async def _get_session_graph(session_id: str = "default") -> str:
        result = await c.memgraph.get_session_graph(session_id)
        return json.dumps(result, ensure_ascii=False)

    # ---- get_corpus_stats -----------------------------------------------
    async def _get_corpus_stats() -> str:
        neo_counts = await c.neo.count_nodes()
        pool = c.pg._pool  # noqa: SLF001
        pg_counts: dict[str, int] = {"chunks": 0, "entities": 0}
        if pool:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT (SELECT count(*) FROM chunks) AS chunks, "
                    "(SELECT count(*) FROM entities) AS entities"
                )
                if rows:
                    pg_counts = {
                        "chunks": int(rows[0]["chunks"]),
                        "entities": int(rows[0]["entities"]),
                    }
        return json.dumps({"neo4j": neo_counts, "pgvector": pg_counts})

    # ---- community_summary ----------------------------------------------
    async def _community_summary(entity_name: str) -> str:
        topic = await c.neo.get_entity_community(entity_name)
        if topic is None:
            return f"No community found for entity: {entity_name}"
        return f"{topic['label']}\n\n{topic['summary']}"

    # ---- list_segments --------------------------------------------------
    async def _list_segments(limit: int = 20) -> str:
        segments = await c.neo.list_segments(limit=limit)
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

    # ---- segment_info ---------------------------------------------------
    async def _segment_info(segment_id: str) -> str:
        info = await c.neo.get_segment_info(segment_id)
        if info is None:
            return f"No SEGMENT found with id: {segment_id}"
        lines = [f"SEGMENT: {segment_id}", ""]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # ---- cypher_query ---------------------------------------------------
    async def _cypher_query(
        template: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        result = await c.neo.run_read_query(template, params or {})
        return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    return [
        GraphKnowsTool(
            name="retrieve",
            description=(
                "Search the knowledge graph. "
                "Modes: HYBRID (default, RRF fusion), SEMANTIC (vector only), "
                "GRAPH_TRAVERSAL (entity hop expansion), SUMMARIES (topic overviews), "
                "FEELING_LUCKY (auto-route). Returns XML <retrieval_context>."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language research question.",
                    },
                    "session_id": {"type": "string", "default": "default"},
                    "search_type": {
                        "type": "string",
                        "enum": [
                            "HYBRID",
                            "SEMANTIC",
                            "GRAPH_TRAVERSAL",
                            "SUMMARIES",
                            "FEELING_LUCKY",
                        ],
                        "default": "HYBRID",
                    },
                    "k_entity": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                    "k_semantic": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
                    "k_final": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
            fn=_retrieve,
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
            fn=_entity_search,
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
            fn=_entity_neighbors,
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
            fn=_get_chunks,
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
            fn=_session_memory,
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
            fn=_ingest,
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
            fn=_poll_job,
            tags=["ingestion"],
        ),
        GraphKnowsTool(
            name="get_session_graph",
            description="Return the entity graph for a session from Memgraph. Shows entities and RELATES edges.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "default": "default"},
                },
                "required": [],
            },
            fn=_get_session_graph,
            tags=["graph", "memory"],
        ),
        GraphKnowsTool(
            name="get_corpus_stats",
            description="Return combined node and row counts from Neo4j and pgvector.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            fn=_get_corpus_stats,
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
            fn=_community_summary,
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
            fn=_list_segments,
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
            fn=_segment_info,
            tags=["admin", "provenance"],
        ),
        GraphKnowsTool(
            name="cypher_query",
            description="Run a parameterized read-only Cypher query against Neo4j. Write keywords are rejected.",
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
            fn=_cypher_query,
            tags=["graph", "admin"],
        ),
    ]
