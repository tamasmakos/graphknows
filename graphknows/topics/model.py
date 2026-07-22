"""Topic records and the result of one topic build."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Topic:
    """One topic: what it is about, what it is made of, and how good it is."""

    topic_id: str
    title: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    # Member entities, ranked by PageRank (highest first).
    entities: list[tuple[str, float]] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    # Mean cosine of members to the topic centroid — how tight the cluster is.
    coherence: float = 0.0
    embedding: list[float] = field(default_factory=list)
    lead_chunk_id: str = ""

    @property
    def size(self) -> int:
        """Member entity count."""
        return len(self.entities)

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """JSON-serialisable view. Embeddings are omitted unless asked for."""
        data = asdict(self)
        data["size"] = self.size
        data["entities"] = [{"name": n, "pagerank": p} for n, p in self.entities]
        if not include_embedding:
            data.pop("embedding", None)
        return data


@dataclass
class TopicSet:
    """Every topic from one build, plus the metrics that grade it."""

    topics: list[Topic] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    namespace: str = ""

    def __len__(self) -> int:
        return len(self.topics)

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """The exported artifact's shape."""
        return {
            "namespace": self.namespace,
            "config": self.config,
            "metrics": self.metrics,
            "topics": [t.to_dict(include_embedding=include_embedding) for t in self.topics],
        }


__all__ = ["Topic", "TopicSet"]
