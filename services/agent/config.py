"""Agent service settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import Field

_GRAPHKNOWS_PKG_DIR = Path(__file__).resolve().parents[2] / "services" / "graphknows"
if str(_GRAPHKNOWS_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPHKNOWS_PKG_DIR))

# graphknows.shared was removed; the base settings now live in graphknows.config.
# Import only the config module directly to avoid heavy container/asyncpg deps.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "graphknows.config",
    _GRAPHKNOWS_PKG_DIR / "graphknows" / "config.py",
)
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules.setdefault("graphknows.config", _mod)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

from graphknows.config import GraphKnowsSettings as BaseServiceSettings  # noqa: E402

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
        default="postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb",  # NOSONAR
        alias="PGVECTOR_DSN",
    )
    memgraph_uri: str = Field(default="bolt://172.18.0.1:17687", alias="MEMGRAPH_URI")  # NOSONAR
    export_graph_html: bool = Field(default=False, alias="EXPORT_GRAPH_HTML")
    graph_html_output_dir: str = Field(
        default="/root/graphknows/output", alias="GRAPH_HTML_OUTPUT_DIR"
    )
    httpx_timeout: float = Field(default=60.0, alias="HTTPX_TIMEOUT")
