"""MCP tool: memory_topics — rebuild the topic layer without re-ingesting."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_runtime


@app.tool()
async def memory_topics(
    namespace: str = "",
    mode: str = "",
    source: str = "community",
    distance: float = 0.35,
    min_size: int = 2,
    n2v_walk_length: int = 20,
    n2v_walks: int = 20,
    n2v_dimensions: int = 32,
    recompute_embeddings: bool = False,
    persist: bool = True,
    export_path: str = "",
) -> dict[str, Any]:
    """Rebuild the topic layer over the already-ingested graph.

    Re-runs clustering, description and grading only — no ingestion, entity
    resolution, PageRank or consolidation — so topics can be swept at different
    settings over an unchanged graph.

    Args:
        namespace: Tenant isolation boundary (default: server's namespace).
        mode: ``extractive`` (deterministic, no LLM), ``llm``, or ``none``.
            Empty uses ``GRAPHKNOWS_TOPICS``.
        source: ``community`` (default) reuses the Louvain partition — near-total
            entity coverage, but no granularity knob, because ArcadeDB's Louvain
            ignores its parameters. ``graph_embedding`` clusters node2vec vectors
            and IS tunable, but only covers entities with >=3 relation edges
            (measured on a conversational graph: 33 of 1049), so check
            ``entity_coverage`` in the result before trusting it.
        distance: Cosine distance at which average-linkage stops merging. THE
            granularity knob — lower gives more, tighter topics.
        min_size: Clusters smaller than this are dropped as stragglers.
        n2v_walk_length: node2vec steps per walk (needs recompute_embeddings).
        n2v_walks: node2vec walks per node (needs recompute_embeddings).
        n2v_dimensions: node2vec dimensionality (needs recompute_embeddings).
        recompute_embeddings: Re-run node2vec with the parameters above before
            clustering. Off by default because it is the expensive part.
        persist: Write the TOPIC layer, replacing the previous build.
        export_path: Optional path for a read-only JSON artifact.

    Returns:
        Build metrics — ``topics``, ``newman_modularity``, ``mean_coherence``,
        ``chunk_coverage``, ``entity_coverage``, ``largest_topic`` — plus
        ``persisted``. Tune against ``newman_modularity`` and
        ``mean_coherence``; the reported ``arcadedb_modularity_untrustworthy``
        rewards coarser partitions and must not be optimised.
    """
    from graphknows.settings import TopicMode
    from graphknows.topics import TopicConfig

    runtime = await get_runtime(namespace)
    resolved = TopicMode(mode) if mode else runtime.settings.topic_mode
    config = TopicConfig(
        source=source if source in ("graph_embedding", "community") else "graph_embedding",
        distance=distance,
        min_size=max(1, min_size),
        mode=resolved,
        n2v_walk_length=n2v_walk_length,
        n2v_walks=n2v_walks,
        n2v_dimensions=n2v_dimensions,
        recompute_embeddings=recompute_embeddings,
    )
    return await runtime.build_topics(  # type: ignore[no-any-return]
        config=config, persist=persist, export_path=export_path
    )
