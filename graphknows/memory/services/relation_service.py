"""RelationService — relation extraction orchestration extracted from Neo4jStore."""

from __future__ import annotations

import logging
from typing import Any

from graphknows.adapters.ports.graph_db import IGraphDb
from graphknows.adapters.ports.llm import ILlm

logger = logging.getLogger(__name__)


class RelationService:
    """Orchestrates DSPy-based relation extraction + typed edge writes.

    Extracted from ``Neo4jStore.extract_and_write_relations``. Accepts
    injected IGraphDb and ILlm ports so it can be tested without
    live database connections.

    Args:
        graph_db: Graph database port (Neo4j-backed in production).
        llm: LLM port (used by DSPyExtractor for relation extraction).
    """

    def __init__(self, graph_db: IGraphDb, llm: ILlm) -> None:
        self._graph_db = graph_db
        self._llm = llm

    async def extract_and_write_relations(
        self,
        stm_data: dict[str, Any],
        openrouter_key: str,
        llm_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    ) -> int:
        """Run DSPyExtractor per chunk; write dynamic typed ENTITY→ENTITY edges.

        Delegates to the underlying IGraphDb implementation which runs
        DSPyExtractor and writes relation edges.

        Args:
            stm_data: Dict produced by MemgraphStore.read_session_subgraph().
            openrouter_key: OpenRouter API key for DSPy LLM calls.
            llm_model: Model identifier for DSPyExtractor.

        Returns:
            Total number of relation edges written.

        Raises:
            Exception: Propagates any extraction or write failure.
        """
        logger.info(
            "RelationService.extract_and_write_relations: %d chunks",
            len(stm_data.get("chunks", [])),
        )
        return await self._graph_db.extract_and_write_relations(
            stm_data, openrouter_key, llm_model
        )
