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

    Topics cluster the Louvain partition: every entity carrying a
    ``community_id`` — in practice all of them. ``min_size`` is applied on top.
    There is no granularity knob at this layer, because ArcadeDB's
    ``algo.louvain`` ignores its parameter map (verified: resolution 0.1 and 5.0
    give identical partitions).

    A second source used to exist here — agglomerative clustering over
    ``ENTITY.graph_embedding`` (node2vec), where a cosine ``distance`` cutoff and
    the walk parameters were real granularity knobs. It was removed for coverage:
    ``compute_graph_embeddings`` skips entities with fewer than 3 relation edges,
    so on a sparse conversational graph it reached 33 of 1049 entities on LoCoMo
    conv-26 — topics describing 3% of the corpus. node2vec itself still runs at
    flush; nothing here reads it.
    """

    # Clustering. ``min_size`` is the only knob: the Louvain partition decides
    # the grouping, and there is no resolution parameter to turn (see above).
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
