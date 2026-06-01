"""EnrichmentService — GDS feature enrichment extracted from Neo4jStore."""

from __future__ import annotations

import logging
from typing import Any

from graphknows.config.adapters.ports.graph_db import IGraphDb
from graphknows.config.adapters.ports.llm import ILlm

logger = logging.getLogger(__name__)


class EnrichmentService:
    """Orchestrates GDS graph data science enrichment pipeline.

    Delegates to ``feature_engineering.run_feature_engineering`` via the
    injected IGraphDb port.  The ILlm dependency is reserved for future
    LLM-assisted community labeling during enrichment.

    Args:
        graph_db: Graph database port (Neo4j-backed in production).
        llm: LLM port (reserved for future use in enrichment commentary).
    """

    def __init__(self, graph_db: IGraphDb, llm: ILlm) -> None:
        self._graph_db = graph_db
        self._llm = llm

    async def run_gds_enrichment(self, session_id: str) -> dict[str, Any]:
        """Run full GDS feature enrichment.

        Delegates to the underlying IGraphDb implementation which calls
        ``feature_engineering.run_feature_engineering``.

        Args:
            session_id: Used to scope projection (session entities only).

        Returns:
            Dict with entity_count, edge_count, community_count, modularity, etc.

        Raises:
            Exception: Propagates any GDS or Neo4j failure.
        """
        logger.info("EnrichmentService.run_gds_enrichment: session_id=%s", session_id)
        return await self._graph_db.run_gds_enrichment(session_id)
