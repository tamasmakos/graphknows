"""Channel base — the symmetric contract every memory signal implements.

A channel is owned end-to-end across the write and read paths:

- ``populate(store, ...)`` — write this channel's structure.
- ``collect(ctx, rt, top_k)`` — return ranked query candidates at retrieval time.

Both hooks have a no-op / empty default, so a read-only channel implements just
``collect`` and a write-only channel just ``populate``. Adding a new signal is
one new subclass plus one line in ``registry.core_channels``, and it is active
on both paths at once.

Candidates are SEGMENTs — the golden unit of retrieval and of provenance —
and every collector reaches them through the store's declared reads, so what
a recall touched is observable (FR-040).
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
        "source_ids",
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
        source_ids: list[str] | None = None,
    ) -> None:
        self.query = query
        self.session_id = session_id
        # The SOURCE ids the session resolved to — a session is a SOURCE, so
        # scoping is by source. ``None`` is namespace-wide.
        self.source_ids = source_ids
        self.emb = emb
        self.emb2 = emb2
        self.words = words
        self.words_graph = words_graph
        self._resolved_entities: list[dict[str, Any]] | None = None

    async def resolve_entities(self, store: Any) -> list[dict[str, Any]]:
        """The entities (``{id, name}``) the query tokens name, cached per query.

        Consumers ask for the same entities from the same ``words_graph``; the
        graph read is issued once and shared, and a run where nothing asks
        never touches the store.
        """
        if self._resolved_entities is None:
            self._resolved_entities = await store.resolve_query_entities(self.words_graph)
        return self._resolved_entities


# Collector return type: segment id -> (fused_score, info_dict).
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
        """Write this channel's graph structure for one session. No-op by default.

        ``store`` is the connected per-namespace GraphStore; its public
        ``command``/``query`` take raw Cypher, so a channel can declare its own
        types (idempotent DDL) and write set-based MERGEs in a handful of
        statements.
        """
        return None

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Return ranked candidates ``{segment_id: (score, info)}``. Empty by default."""
        return {}


__all__ = ["Channel", "ChannelContext", "CollectResult"]
