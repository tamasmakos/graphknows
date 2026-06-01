"""TopicService — topic summarization extracted from Neo4jStore."""

from __future__ import annotations

import logging

from graphknows.config.adapters.ports.graph_db import IGraphDb
from graphknows.config.adapters.ports.llm import ILlm
from graphknows.config.adapters.ports.vector_db import IVectorDb

logger = logging.getLogger(__name__)


class TopicService:
    """Orchestrates Leiden community detection + LLM topic summarization.

    Extracted from ``Neo4jStore.summarize_topics``. The LLM step can be
    bypassed by providing an ILlm implementation that returns empty strings.

    Args:
        graph_db: Graph database port (Neo4j-backed in production).
        llm: LLM port used for topic title/summary generation.
    """

    def __init__(self, graph_db: IGraphDb, llm: ILlm) -> None:
        self._graph_db = graph_db
        self._llm = llm

    async def summarize_topics(
        self,
        session_id: str,
        openrouter_key: str,
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
        pg_store: IVectorDb | None = None,
    ) -> int:
        """Detect entity communities, summarize with LLM, write TOPICs.

        Delegates to the underlying IGraphDb implementation which runs the
        Leiden algorithm and writes TOPIC nodes.

        Args:
            session_id: The session to summarize.
            openrouter_key: OpenRouter API key. If empty, LLM step is skipped.
            llm_model: Model identifier passed to OpenRouter.
            pg_store: Optional PGVectorStore — if provided, writes ltm_topics rows.

        Returns:
            Number of topics created.
        """
        logger.info("TopicService.summarize_topics: session_id=%s", session_id)
        return await self._graph_db.summarize_topics(
            session_id, openrouter_key, llm_model, pg_store
        )
