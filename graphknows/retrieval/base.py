"""BaseGraphRetriever — shared abstract base for all retrieval modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphknows.retrieval.models.context import RetrievalContext


class BaseGraphRetriever(ABC):
    """Abstract base for retrieval over graphknows memory stores.

    Both deterministic (DETRetriever) and agentic (AgenticRetriever)
    modes inherit from this class, ensuring a uniform interface regardless
    of whether an LLM is involved in retrieval.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        scope: str = "both",
    ) -> RetrievalContext:
        """Retrieve relevant context for *query*.

        Args:
            query: Natural-language question.
            session_id: Session namespace for STM scoping.
            top_k: Maximum results to return.
            scope: ``"stm"``, ``"ltm"``, or ``"both"`` (default).
        """
        ...
