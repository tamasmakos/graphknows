"""A session-bound view over :class:`~graphknows.memory.Memory` for LangGraph.

The rest of ``graphknows.integrations`` is framework-agnostic: ``recall`` and
``remember`` in ``_langgraph_hooks.py`` are plain ``state -> partial-state``
callables, bound to a memory and a session via ``functools.partial``. That is
flexible but verbose for the common case of "one session, one memory, three
node callables". ``GraphKnowsMemory`` is that common case wrapped up — it
owns (or borrows) a ``Memory`` and hands out bound node methods, so wiring a
graph is three ``add_node`` calls instead of three ``partial(...)`` calls.

It depends on nothing from langgraph — its nodes are plain dict-in/dict-out
callables — so importing this module is free, exactly like the hooks module
it wraps.
"""

from __future__ import annotations

from typing import Any

from graphknows.integrations._langgraph_hooks import recall as _recall
from graphknows.integrations._langgraph_hooks import remember as _remember
from graphknows.memory import Memory


class GraphKnowsMemory:
    """Session-bound memory with ready-to-use LangGraph node callables.

    Builds its own :class:`Memory` when none is supplied, and only closes
    that self-built instance on ``__aexit__`` — closing a caller-supplied
    ``Memory`` on exit would be a surprise for a caller who wants to reuse it
    across sessions.
    """

    def __init__(
        self,
        session_id: str,
        *,
        memory: Memory | None = None,
        namespace: str | None = None,
        top_k: int = 5,
        messages_key: str = "messages",
        out_key: str = "memories",
        cross_session: bool = True,
    ) -> None:
        # No user_id/agent_id here on purpose. Memory.add/search accept them to
        # DERIVE a session id, but this class is handed one explicitly, so they
        # would have nothing left to do — and a parameter that is accepted and
        # then ignored is indistinguishable from one that works. Isolation
        # between users is the namespace.
        self._session_id = session_id
        self._cross_session = cross_session
        self._top_k = top_k
        self._messages_key = messages_key
        self._out_key = out_key
        self._owns_memory = memory is None
        self._memory = memory if memory is not None else Memory(namespace=namespace)

    @property
    def memory(self) -> Memory:
        """The underlying ``Memory`` instance (read-only)."""
        return self._memory

    @property
    def session_id(self) -> str:
        """The session this view is bound to (read-only)."""
        return self._session_id

    async def __aenter__(self) -> GraphKnowsMemory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        # Only close a Memory this object built — a caller-supplied Memory
        # may be shared across sessions and outlives this context.
        if self._owns_memory:
            await self._memory.close()

    async def recall(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node: recall memories for the latest turn.

        Delegates to the framework-agnostic ``recall`` hook bound to this
        session's settings, so the message-extraction logic (including
        LangChain multimodal content blocks) lives in exactly one place.
        """
        return await _recall(
            state,
            memory=self._memory,
            session_id=self._session_id,
            messages_key=self._messages_key,
            top_k=self._top_k,
            out_key=self._out_key,
            cross_session=self._cross_session,
        )

    async def remember(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node: buffer the latest turn into short-term memory."""
        return await _remember(
            state,
            memory=self._memory,
            session_id=self._session_id,
            messages_key=self._messages_key,
        )

    async def flush(self) -> dict[str, Any]:
        """Ingest this session's buffered turns and consolidate the graph."""
        return await self._memory.flush(session_id=self._session_id)

    async def stats(self) -> dict[str, Any]:
        """This session's memory statistics (turns/chunks/entities/topics)."""
        return await self._memory.stats(self._session_id)
