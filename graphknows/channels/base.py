"""Channel base — the symmetric contract every memory signal implements.

A channel is owned end-to-end across the write and read paths:

- ``ingest(rt, ...)`` — derive + write this channel's structure for one chunk.
- ``collect(ctx, rt, top_k)`` — return ranked query candidates at retrieval time.

Both hooks have a no-op / empty default, so a read-only channel implements just
``collect`` and a write-only channel just ``ingest``. Adding a new signal — e.g.
metadata extraction — is one new subclass plus one line in
``registry.core_channels``, and it is active on both paths at once.

There is no separate promote hook: the golden layer consolidates a session in
place rather than copying it into a second database, so everything a channel
writes is written on the ingest path.
"""

from __future__ import annotations

from typing import Any


class ChannelContext:
    """Per-query shared state passed to every channel.

    Eager fields are computed by the retriever spine before fan-out;
    ``resolve_entities`` is lazy, so a run whose channels never ask for entity
    names never touches the store.
    """

    __slots__ = (
        "_resolved_entities",
        "emb",
        "emb2",
        "query",
        "session_id",
        "state",
        "words",
        "words_graph",
    )

    def __init__(
        self,
        query: str,
        session_id: str,
        emb: list[float],
        emb2: list[float] | None,
        words: list[str],
        words_graph: list[str],
        state: str | None = None,
    ) -> None:
        self.query = query
        self.session_id = session_id
        # Lifecycle filter derived from the caller's scope (raw / consolidated /
        # None=both). The raw scope itself is not carried: nothing read it.
        self.state = state
        self.emb = emb
        self.emb2 = emb2
        self.words = words
        self.words_graph = words_graph
        self._resolved_entities: list[str] | None = None

    async def resolve_entities(self, store: Any) -> list[str]:
        """Entity names the query tokens resolve to (cached per query).

        Consumers ask for the same names from the same ``words_graph``; the
        graph read is issued once and shared, and a run where nothing asks
        never touches the store.
        """
        if self._resolved_entities is None:
            self._resolved_entities = await store.resolve_query_entities(self.words_graph)
        return self._resolved_entities


# Collector return type: chunk_id -> (fused_score, info_dict).
CollectResult = dict[str, tuple[float, dict[str, Any]]]


class Channel:
    """One memory signal — a core collector, or one a caller supplies.

    Subclasses set ``name`` (the sources label) and implement ``collect``.

    There is no ``profiles`` class var and no ``enable_x`` gate: the core set
    is fixed (``registry.core_channels``), and anything else is passed in by
    the caller or, from the dialogue pack on, by a pack.
    """

    name: str = "channel"

    async def populate(self, store: Any, session_id: str) -> None:
        """Write this channel's graph structure for one flushed session. No-op by default.

        Called once per :meth:`graphknows.Memory.flush_memory`, AFTER the
        session's buffered turns have been drained into CHUNK vertices and
        before the analytics stages — the one moment custom edges to those
        chunks can actually land on something. ``store`` is the connected
        per-namespace GraphStore; its public ``command``/``query`` take raw
        Cypher, so a channel can declare its own types (idempotent DDL) and
        write set-based MERGEs in a handful of statements.

        This replaces the per-chunk ``ingest`` hook, which was declared,
        documented, and never called by anything — and whose shape was wrong
        anyway: session-level structure (one edge per session) written per
        chunk is N redundant round-trips for one write.
        """
        return None

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Return ranked candidates ``{chunk_id: (score, info)}``. Empty by default."""
        return {}


__all__ = ["Channel", "ChannelContext", "CollectResult"]
