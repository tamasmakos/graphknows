"""A session-bound view over :class:`~graphknows.memory.Memory` for LangGraph.

The rest of this adapter is framework-agnostic: ``recall`` and
``remember`` in ``_hooks.py`` are plain ``state -> partial-state``
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

import logging
from collections.abc import Callable
from typing import Any

from graphknows.exceptions import ConfigurationError
from graphknows.integrations.langgraph._hooks import _last_text, resolve_infer
from graphknows.integrations.langgraph._hooks import recall as _recall
from graphknows.integrations.langgraph._hooks import remember as _remember
from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings

logger = logging.getLogger(__name__)


class GraphKnowsMemory:
    """Session-bound memory with ready-to-use LangGraph node callables.

    Builds its own :class:`Memory` from ``namespace``/``settings`` when none is
    supplied, and only closes that self-built instance on ``__aexit__`` —
    closing a caller-supplied ``Memory`` on exit would be a surprise for a
    caller who wants to reuse it across sessions. Configuring an integration is
    therefore one argument, not a construction ritual::

        async with GraphKnowsMemory("s1", namespace="demo") as mem:
            await mem.add("Mochi is a ragdoll", speaker="Gina")
            await mem.flush()
    """

    def __init__(
        self,
        session_id: str,
        *,
        memory: Memory | None = None,
        namespace: str | None = None,
        settings: GraphKnowsSettings | None = None,
        channels: list[Any] | None = None,
        top_k: int = 5,
        messages_key: str = "messages",
        out_key: str = "memories",
        cross_session: bool = True,
        infer: bool | Callable[[str], bool] = True,
    ) -> None:
        # No user_id/agent_id here on purpose. Memory.add/search accept them to
        # DERIVE a session id, but this class is handed one explicitly, so they
        # would have nothing left to do — and a parameter that is accepted and
        # then ignored is indistinguishable from one that works. Isolation
        # between users is the namespace.
        if memory is not None and (
            namespace is not None or settings is not None or channels is not None
        ):
            # These configure the Memory this class would build. Handed one
            # already built, they have nothing to configure — and silently
            # dropping them sent a caller's turns to the default namespace with
            # no error, which reads as "memory is empty" three steps later.
            raise ConfigurationError(
                "GraphKnowsMemory: pass either memory=, or namespace=/settings=/channels= "
                "— not both. A supplied Memory already carries its own namespace, "
                "settings and channels."
            )
        self._session_id = session_id
        self._cross_session = cross_session
        self._top_k = top_k
        # The extraction policy, asked once per turn on both write paths. A flat
        # bool, or a predicate asked per turn. Per-turn control used to mean
        # abandoning `remember` and driving `ingest_memory` by hand.
        self._infer = infer
        self._messages_key = messages_key
        self._out_key = out_key
        self._owns_memory = memory is None
        self._memory = (
            memory
            if memory is not None
            else Memory(settings, namespace=namespace, channels=channels)
        )
        # Turns buffered since the last flush. A buffered turn mints nothing
        # until flush drains it, so exiting with a non-zero count is a silently
        # empty graph — see __aexit__.
        self._unflushed = 0

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
        # Exiting without flushing is not an error — a caller may deliberately
        # leave the buffer for a later session to drain — but it is
        # indistinguishable from a broken install unless it is said out loud:
        # buffered turns are retrievable only via scope="stm", mint no entities,
        # and answer nothing from the graph.
        if self._unflushed:
            logger.warning(
                "GraphKnowsMemory(%r) closed with %d unflushed turn(s): they mint no "
                "entities and are invisible to graph recall until flush() runs.",
                self._session_id,
                self._unflushed,
            )
        # Only close a Memory this object built — a caller-supplied Memory
        # may be shared across sessions and outlives this context.
        if self._owns_memory:
            await self._memory.close()

    async def add(
        self,
        text: str,
        *,
        speaker: str = "",
        timestamp: str = "",
        role: str = "user",
        infer: bool | Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """Buffer one turn into this session's short-term memory.

        The direct write, for a caller that already has a structured turn —
        ``remember`` is the same operation reading a framework's message state.
        Each field goes where it belongs instead of being flattened into the
        text: ``speaker`` becomes the turn's name (and the chunk's speaker),
        ``timestamp`` becomes the anchor its relative dates resolve against.
        Flattening either one into ``text`` is what mints greetings like
        "Hey Jon" as entities and dates a 2023 conversation to ingestion time.

        ``infer`` is the extraction decision for this turn: a turn stored with
        ``infer=False`` is still written, embedded and retrievable as text — it
        just mints no entities. Omit it to use the session's policy (the
        ``infer=`` given to ``__init__``, which may be a per-turn predicate);
        pass a bool or a predicate to override it for this turn alone.

        Buffered turns reach the graph on :meth:`flush`, not here.
        """
        body = text.strip()
        if not body:
            return {"turns": 0}
        turn: dict[str, Any] = {"role": role, "content": body}
        if speaker:
            turn["name"] = speaker
        if timestamp:
            turn["timestamp"] = timestamp
        result = await self._memory.ingest_memory(
            [turn],
            session_id=self._session_id,
            infer=resolve_infer(self._infer if infer is None else infer, body),
        )
        self._unflushed += 1
        return result

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
        update = await _remember(
            state,
            memory=self._memory,
            session_id=self._session_id,
            messages_key=self._messages_key,
            infer=self._infer,
        )
        # The hook is a no-op on a turn with no text, so counting node calls
        # instead of stored turns would warn "N unflushed turns" over an empty
        # buffer — a false alarm on exactly the signal this counter exists for.
        # Asking the hook's own predicate keeps the two from drifting apart.
        if _last_text(state, self._messages_key):
            self._unflushed += 1
        return update

    async def flush(self) -> dict[str, Any]:
        """Ingest this session's buffered turns and consolidate the graph."""
        result = await self._memory.flush()
        self._unflushed = 0
        return result

    async def stats(self) -> dict[str, Any]:
        """This session's memory statistics (turns/chunks/entities/topics)."""
        return await self._memory.stats(self._session_id)
