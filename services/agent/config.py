"""Agent service settings."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import Field

_SHARED_PKG_DIR = Path(__file__).resolve().parents[2] / "libs" / "python" / "graphknows-shared"
if str(_SHARED_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_PKG_DIR))

from graphknows.shared.config import BaseServiceSettings

SESSION_GRAPH_BACKEND: str = os.getenv("SESSION_GRAPH_BACKEND", "memgraph")


class AgentSettings(BaseServiceSettings):
    """Settings for the Graphknows agent service.

    Attributes:
        graphknows_url: Base URL of the graphknows unified memory service.
        search_url: Base URL of the search service.
        memgraph_uri: Bolt URI for the ephemeral Memgraph session store.
        httpx_timeout: Timeout in seconds for downstream HTTP calls.
    """
    graphknows_url: str = Field(default="http://localhost:8025", alias="GRAPHKNOWS_URL")
    graphknows_mode: str = Field(default="http", alias="GRAPHKNOWS_MODE")
    search_url: str = Field(default="http://localhost:8000", alias="SEARCH_URL")
    pgvector_dsn: str = Field(
        default="postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb",
        alias="PGVECTOR_DSN",
    )
    memgraph_uri: str = Field(default="bolt://172.18.0.1:17687", alias="MEMGRAPH_URI")
    export_graph_html: bool = Field(default=False, alias="EXPORT_GRAPH_HTML")
    graph_html_output_dir: str = Field(default="/root/graphknows/output", alias="GRAPH_HTML_OUTPUT_DIR")
    httpx_timeout: float = Field(default=60.0, alias="HTTPX_TIMEOUT")
