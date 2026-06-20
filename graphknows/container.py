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
import logging
from typing import Any

from graphknows._container_tools import _build_tools
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

    @classmethod
    def from_env(cls) -> GraphKnowsContainer:
        """Build a container from environment variables / .env file."""
        return cls(GraphKnowsSettings())

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
