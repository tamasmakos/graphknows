"""Channel base model — the contract every retrieval signal implements.

``ChannelContext`` holds per-query shared state, computed once and passed to every
channel; expensive derived fields (frame parse, spaCy doc) are lazy so a channel
that never asks for them never pays for them. ``Channel`` is the end-to-end
contract: schema DDL + ingest write + flush promotion + query collection. Every
method has a no-op default, so a pure-retrieval channel implements only
``collect`` and a pure-ingest channel only ``ingest``/``promote``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ChannelStores:
    """Shared store handle passed to ingest/promote hooks (``rt``).

    The DETRetriever itself satisfies the same duck-typed contract on the query
    side, so ``collect`` receives the retriever and ``promote``/``ingest`` receive
    this handle — both expose ``mg``/``pg``/``neo``/``embed_model``.
    """

    mg: Any = None
    pg: Any = None
    neo: Any = None
    embed_model: str = ""


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
    ) -> None:
        self.query = query
        self.session_id = session_id
        self.scope = scope
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
            from graphknows.extraction.frames import parse_query
            self._frame_parse = parse_query(self.query)
        return self._frame_parse


# Collector return type: chunk_id -> (fused_score, info_dict).
CollectResult = dict[str, tuple[float, dict[str, Any]]]


class Channel:
    """One retrieval signal, owned across ingest, flush, and query.

    Subclasses set ``name`` (the trace/sources label) and ``profiles`` (which
    memory modes it runs in), plus whichever of the three hooks it needs. The
    ``rt`` handle passed to the hooks exposes the shared stores (``rt.mg``,
    ``rt.pg``, ``rt.neo``, ``rt.embed_model``) and, for the transitional adapter
    channels, the retriever's collector methods.
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

    async def promote(self, rt: Any, session_id: str, record: Any) -> None:
        """Promote/derive this channel's LTM structure once per flush. No-op by default."""
        return None

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Return ranked candidates ``{chunk_id: (score, info)}``. Empty by default."""
        return {}
