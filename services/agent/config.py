"""Agent service settings."""
from __future__ import annotations

from pydantic import Field
from graphknows.shared.config import BaseServiceSettings


class AgentSettings(BaseServiceSettings):
    """Settings for the Graphknows agent service.

    Attributes:
        graphrag_url: Base URL of the graphrag retrieval service.
        graphgen_url: Base URL of the graphgen pipeline service.
        search_url: Base URL of the search service.
        memgraph_uri: Bolt URI for the ephemeral Memgraph session store.
        httpx_timeout: Timeout in seconds for downstream HTTP calls.
    """
    graphrag_url: str = Field(default="http://localhost:8001", alias="GRAPHRAG_URL")
    graphgen_url: str = Field(default="http://localhost:8003", alias="GRAPHGEN_URL")
    search_url: str = Field(default="http://localhost:8004", alias="SEARCH_URL")
    pgvector_dsn: str = Field(
        default="postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb",
        alias="PGVECTOR_DSN",
    )
    memgraph_uri: str = Field(default="bolt://172.17.0.1:17687", alias="MEMGRAPH_URI")
    httpx_timeout: float = Field(default=60.0, alias="HTTPX_TIMEOUT")
