"""LTM enrichment result models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CentralityResult(BaseModel):
    """Result of centrality computation (PageRank, betweenness, degree)."""

    entity_count: int = 0
    edge_count: int = 0
    error: str = ""


class CommunityResult(BaseModel):
    """Result of community detection."""

    community_count: int = 0
    modularity: float = 0.0
    modularities: list[float] = Field(default_factory=list)
    ran_iterations: int = 0
    did_converge: bool = False
    changed_community_ids: list[int] = Field(default_factory=list)
    algorithm_used: str = ""
    entity_count: int = 0
    edge_count: int = 0
    silhouette_cosine: float | None = None
    global_separation: float | None = None
    global_overlap: float | None = None
    error: str = ""


class GDSResult(BaseModel):
    """Combined result of a full GDS enrichment run."""

    entity_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    modularity: float = 0.0
    modularities: list[float] = Field(default_factory=list)
    ran_iterations: int = 0
    did_converge: bool = False
    changed_community_ids: list[int] = Field(default_factory=list)
    silhouette_cosine: float | None = None
    global_separation: float | None = None
    global_overlap: float | None = None
    unweighted_modularity: float | None = None
    error: str = ""
