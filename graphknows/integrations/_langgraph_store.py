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

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from graphknows.models.hit import Hit

logger = logging.getLogger(__name__)

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


# `BaseStore` resolves to Any when langgraph is absent (the MissingExtraError
# guard above), so mypy cannot see a real base class here.
class GraphKnowsStore(BaseStore):  # type: ignore[misc]
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

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async dispatch of langgraph store operations onto graphknows verbs."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, PutOp):
                # A langgraph PutOp result is None by contract; `_put` returns
                # nothing, so append the None rather than the call expression.
                await self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(await self._search(op))
            else:
                # GetOp / ListNamespacesOp: the graph-memory model has no
                # key-addressable fetch or namespace enumeration.
                results.append(None if isinstance(op, GetOp) else [])
        return results

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Sync batch is unsupported — GraphKnowsStore is async-only."""
        raise NotImplementedError(
            "GraphKnowsStore is async-only; use the async graph API (abatch)."
        )

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
            if not isinstance(hit, Hit):
                # Broken wiring, not a variation worth tolerating quietly: a
                # silent skip is indistinguishable from an empty store.
                logger.warning(
                    "GraphKnowsStore.search: skipped a recall result of type %s; "
                    "the memory is not returning Hit objects.",
                    type(hit).__name__,
                )
                continue
            # created_at/updated_at are REQUIRED positional fields on SearchItem;
            # omitting them raised TypeError on every search() call. It went
            # unseen because langgraph is an optional extra that is absent from
            # the environment the gates run in, so --ignore-missing-imports
            # resolved SearchItem to Any and mypy stayed silent. graphknows does
            # not track per-hit write times, so both carry the retrieval time —
            # the honest value for a view materialised from the graph.
            now = datetime.now(UTC)
            items.append(
                SearchItem(
                    namespace=op.namespace_prefix,
                    key=str(hit.metadata.get("langgraph_key") or i),
                    # The whole hit, not a two-field excerpt: dropping speaker
                    # and ts here left a langgraph agent unable to date or
                    # attribute anything it read back out of the store.
                    value=hit.to_dict(),
                    created_at=now,
                    updated_at=now,
                    score=hit.score,
                )
            )
        return items


__all__ = ["GraphKnowsStore", "Item"]
