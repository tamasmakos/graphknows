"""Channel base — the symmetric contract every memory signal implements.

A channel is owned end-to-end across the write and read paths:

- ``ingest(rt, ...)`` — derive + write this channel's structure for one chunk.
- ``collect(ctx, rt, top_k)`` — return ranked query candidates at retrieval time.

Both hooks have a no-op / empty default, so a read-only channel implements just
``collect`` and a write-only channel just ``ingest``. Adding a new signal — e.g.
metadata extraction — is one new subclass plus one ``register()`` call, and it
is active on both paths at once.

There is no separate promote hook: the golden layer consolidates a session in
place rather than copying it into a second database, so everything a channel
writes is written on the ingest path.
"""

from __future__ import annotations

from typing import Any, ClassVar


class ChannelContext:
    """Per-query shared state passed to every channel.

    Eager fields are computed by the retriever spine before fan-out;
    ``frame_parse`` is lazy (only frame-aware channels trigger it, so non-frame
    runs never load spaCy/FrameNet).
    """

    __slots__ = (
        "_frame_parse",
        "emb",
        "emb2",
        "query",
        "scope",
        "session_id",
        "state",
        "words",
        "words_graph",
    )

    def __init__(
        self,
        query: str,
        session_id: str,
        scope: str,
        emb: list[float],
        emb2: list[float] | None,
        words: list[str],
        words_graph: list[str],
        state: str | None = None,
    ) -> None:
        self.query = query
        self.session_id = session_id
        self.scope = scope
        # Lifecycle filter derived from scope (raw / consolidated / None=both).
        self.state = state
        self.emb = emb
        self.emb2 = emb2
        self.words = words
        self.words_graph = words_graph
        self._frame_parse: dict[str, Any] | None = None

    def frame_parse(self) -> dict[str, Any]:
        """Parse the query into frames/anchors (cached).

        Loads spaCy+FrameNet on first use — only invoked by frame-aware channels.
        """
        if self._frame_parse is None:
            from graphknows.channels._framenet import parse_query

            self._frame_parse = parse_query(self.query)
        return self._frame_parse


# Collector return type: chunk_id -> (fused_score, info_dict).
CollectResult = dict[str, tuple[float, dict[str, Any]]]


class Channel:
    """One memory signal, owned across ingest, flush, and query.

    Subclasses set ``name`` (the trace/sources label) and ``profiles`` (which
    memory modes it runs in), plus whichever of the three hooks it needs.
    """

    name: str = "channel"
    profiles: ClassVar[frozenset[str]] = frozenset({"llm_free", "llm_assisted"})
    stm_ddl: ClassVar[list[tuple[str, str]]] = []
    ltm_ddl: ClassVar[list[tuple[str, str]]] = []

    async def ingest(
        self, rt: Any, session_id: str, chunk_id: str, text: str, extraction: Any
    ) -> None:
        """Derive + write this channel's STM structure for one chunk. No-op by default."""
        return None

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Return ranked candidates ``{chunk_id: (score, info)}``. Empty by default."""
        return {}


__all__ = ["Channel", "ChannelContext", "CollectResult"]
