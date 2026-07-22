"""Tunable settings for the topic stage.

Split from the memory-wide settings because topics are meant to be *re-run* at
different settings over an unchanged graph. Everything that changes the shape of
the output lives here, so one object identifies one topic build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from graphknows.settings import TopicMode


@dataclass(frozen=True)
class TopicConfig:
    """One topic build's parameters.

    ``source`` decides what gets clustered, and the two options trade coverage
    against tunability:

    - ``community`` (default): reuse the Louvain partition. Covers every entity
      that has a ``community_id`` — in practice all of them — but offers no
      granularity knob, because ArcadeDB's ``algo.louvain`` ignores its
      parameter map (verified: resolution 0.1 and 5.0 give identical
      partitions). ``min_size`` is still applied.
    - ``graph_embedding``: cluster node2vec entity vectors, where ``distance``
      and the node2vec parameters below become real granularity knobs. The
      catch is coverage: ``compute_graph_embeddings`` skips entities with fewer
      than 3 relation edges, so on a sparse conversational graph it may reach
      only a few percent of entities (measured on LoCoMo conv-26: 33 of 1049).
      Check ``entity_coverage`` in the build metrics before trusting it, and
      prefer it on graphs with dense relation structure.

    Whichever source is used, the node2vec parameters never affect the Louvain
    community layer — nothing feeds node2vec output back into clustering there.
    """

    source: str = "community"  # community | graph_embedding

    # node2vec — the walk parameters that shape the embedding space. Hardcoded in
    # the store before this stage existed; surfaced here so a sweep can move them.
    n2v_walk_length: int = 20
    n2v_walks: int = 20
    n2v_dimensions: int = 32
    recompute_embeddings: bool = False  # re-run node2vec before clustering

    # Clustering. ``distance`` is the granularity knob: cosine distance at which
    # average-linkage stops merging. Lower -> more, tighter topics.
    distance: float = 0.35
    min_size: int = 2  # clusters smaller than this are stragglers, not topics

    # Generation
    mode: TopicMode = TopicMode.extractive
    max_title_entities: int = 3
    max_passages: int = 12
    summary_max_chars: int = 400
    keywords_per_topic: int = 8

    # Provenance, so an exported artifact says which build produced it.
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (enums flattened to their values)."""
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    def fingerprint(self) -> str:
        """Stable id for this parameter set — distinguishes runs in an export."""
        import hashlib
        import json

        payload = json.dumps(
            {k: v for k, v in self.to_dict().items() if k not in ("label", "extra")},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


__all__ = ["TopicConfig"]
