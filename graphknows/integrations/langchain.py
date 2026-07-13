"""LangChain integration — a retriever that reads from graphknows memory.

``GraphKnowsRetriever`` wraps a :class:`~graphknows.memory.Memory` (or the
out-of-process ``RemoteMemory``) as a ``langchain_core`` retriever, so a chain can
pull scoped memory as ``Document`` results. ``langchain_core`` is imported lazily
inside the factory, so importing this module never requires langchain.
"""

from __future__ import annotations

from typing import Any


def build_retriever(
    memory: Any,
    *,
    user_id: str = "",
    agent_id: str = "",
    run_id: str = "",
    top_k: int = 5,
    scope: str = "both",
) -> Any:
    """Return a ``langchain_core`` ``BaseRetriever`` backed by graphknows memory.

    Raises ``MissingExtraError`` guidance if ``langchain-core`` is not installed.
    """
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ModuleNotFoundError as exc:  # pragma: no cover - optional integration
        raise ImportError(
            "GraphKnowsRetriever requires langchain-core. Install it with "
            "'pip install langchain-core'."
        ) from exc

    from graphknows._async_bridge import _run as _bg_run

    class GraphKnowsRetriever(BaseRetriever):  # type: ignore[misc]
        """LangChain retriever backed by a graphknows Memory scope."""

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            result = _bg_run(
                memory.search(
                    query,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    top_k=top_k,
                    scope=scope,
                )
            )
            return [
                Document(page_content=hit.get("text", ""), metadata=hit.get("metadata", {}))
                for hit in result.get("hits", [])
            ]

    return GraphKnowsRetriever()


__all__ = ["build_retriever"]
