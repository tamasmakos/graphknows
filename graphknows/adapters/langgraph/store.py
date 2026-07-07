"""GraphKnowsStore — a langgraph ``BaseStore`` backed by graphknows memory.

Maps langgraph's store protocol onto the graphknows verbs:

- ``put(namespace, key, value)``  → ``ingest_memory`` (namespace → session id)
- ``search(namespace, query=...)`` → ``recall_memory`` (hits → SearchItem)
- ``get`` / ``list_namespaces``    → not supported by the graph-memory model

This module requires the ``langgraph`` extra (``pip install graphknows[langgraph]``);
importing it without langgraph raises :class:`MissingExtraError`. It is validated
by ``examples/langgraph_agent.py`` against a live ArcadeDB.
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        Item,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without langgraph
    from graphknows.exceptions import MissingExtraError

    raise MissingExtraError("The LangGraph adapter", "langgraph") from exc


def _session_id(namespace: tuple[str, ...]) -> str:
    """Flatten a langgraph namespace tuple into a graphknows session id."""
    return "/".join(namespace) if namespace else "default"


class GraphKnowsStore(BaseStore):
    """A langgraph ``BaseStore`` whose long-term memory is a graphknows backend.

    Args:
        memory: A graphknows :class:`~graphknows.Memory` (or any object exposing
            ``ingest_memory`` / ``recall_memory``). Construct one from settings and
            reuse it across the graph.
        top_k: Default number of hits returned by ``search`` when ``limit`` is
            not otherwise constrained.
    """

    def __init__(self, memory: Any, *, top_k: int = 5) -> None:
        self._memory = memory
        self._top_k = top_k

    async def abatch(self, ops: list[Op]) -> list[Result]:
        """Async dispatch of langgraph store operations onto graphknows verbs."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                results.append(await self._put(op))
            elif isinstance(op, SearchOp):
                results.append(await self._search(op))
            else:
                # GetOp / ListNamespacesOp: the graph-memory model has no
                # key-addressable fetch or namespace enumeration.
                results.append(None if isinstance(op, GetOp) else [])
        return results

    def batch(self, ops: list[Op]) -> list[Result]:
        """Sync batch is unsupported — GraphKnowsStore is async-only."""
        raise NotImplementedError("GraphKnowsStore is async-only; use the async graph API (abatch).")

    async def _put(self, op: PutOp) -> None:
        if op.value is None:
            # A None value is a delete in langgraph; graphknows purges by session.
            return None
        text = op.value.get("text") if isinstance(op.value, dict) else str(op.value)
        if not text:
            return None
        await self._memory.ingest_memory(
            str(text),
            session_id=_session_id(op.namespace),
            metadata={"langgraph_key": op.key},
        )
        return None

    async def _search(self, op: SearchOp) -> list[SearchItem]:
        query = getattr(op, "query", None)
        if not query:
            return []
        session_id = _session_id(op.namespace_prefix)
        limit = getattr(op, "limit", None) or self._top_k
        result = await self._memory.recall_memory(
            query, session_id=session_id, top_k=limit, scope="both"
        )
        hits = result.get("hits", []) if isinstance(result, dict) else []
        items: list[SearchItem] = []
        for i, hit in enumerate(hits):
            if not isinstance(hit, dict):
                continue
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=str(hit.get("metadata", {}).get("langgraph_key") or i),
                    value={"text": hit.get("text", ""), "sources": hit.get("sources", "")},
                    score=float(hit.get("score", 0.0) or 0.0),
                )
            )
        return items


__all__ = ["GraphKnowsStore", "Item"]
