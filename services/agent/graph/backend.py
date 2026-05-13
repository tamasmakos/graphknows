from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EntityFeatures:
    entity_id: str = ""
    pagerank: float = 0.0
    betweenness: float = 0.0
    community: int = -1
    fastrp_embedding: list[float] = field(default_factory=list)


@dataclass
class NodeFeatures:
    by_entity_id: dict[str, EntityFeatures] = field(default_factory=dict)

    def top_n_by_pagerank(self, n: int = 10) -> list[EntityFeatures]:
        """Return top-N EntityFeatures sorted by PageRank descending."""
        return sorted(
            self.by_entity_id.values(),
            key=lambda ef: ef.pagerank,
            reverse=True,
        )[:n]


class GraphBackend(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def write_entities(self, session_id: str, entities: list[Any]) -> None: ...

    async def write_relations(self, session_id: str, relations: list[Any]) -> None: ...

    async def two_hop_neighbours(self, session_id: str, entity_ids: list[str]) -> list[str]: ...

    async def delete_session(self, session_id: str) -> None: ...

    async def compute_node_features(self, session_id: str) -> NodeFeatures: ...
