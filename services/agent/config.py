"""Agent service settings."""
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field

_SHARED_PKG_DIR = Path(__file__).resolve().parents[2] / "packages" / "shared-py"
if str(_SHARED_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_PKG_DIR))

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
    memgraph_uri: str = Field(default="bolt://172.17.0.1:17687", alias="MEMGRAPH_URI")
    httpx_timeout: float = Field(default=60.0, alias="HTTPX_TIMEOUT")
