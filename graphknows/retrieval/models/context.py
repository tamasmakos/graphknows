"""RetrievalContext — what the retriever hands back.

Plain dicts, no wire format: consumers (``memory.py``, the MCP ``memory_query``
tool) read ``fused_results`` directly. An XML/dict serialiser with
``graph_matches`` / ``semantic_matches`` sections used to live here; both
sections were permanently ``count="0"`` because the retriever never populated
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalContext:
    """A retrieval result.

    Attributes:
        query: Original query string.
        session_id: Session namespace used for vector search.
        fused_results: RRF-fused deduped chunk dicts, sorted by ``rrf_score``
            descending, each carrying ``chunk_id``, ``doc_id``, ``text``,
            ``metadata``, ``speaker``, ``ts``, ``rrf_score``, ``sources``,
            ``rank`` and ``entities``.
        search_type: Which retriever answered.
        facts: Structured fact-sheet (D8): compact strings of the query
            entities' typed relations + frame-instance role fills, fed to the
            generator alongside the passages. Empty when
            ``GRAPHKNOWS_FACT_CONTEXT`` is off.
    """

    query: str
    session_id: str
    fused_results: list[dict] = field(default_factory=list)
    search_type: str = "HYBRID"
    facts: list[str] = field(default_factory=list)
