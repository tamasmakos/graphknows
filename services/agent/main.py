"""Graphknows Agent Service entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from agent.config import AgentSettings
from agent.session.manager import SessionGraphManager
from agent.router import _init_router, router
from services.agent.config import SESSION_GRAPH_BACKEND
from services.agent.graph.backends.memgraph import MemgraphBackend
from services.agent.graph.backends.neo4j_gds import Neo4jGDSBackend
from services.agent.tools.graph.client import GraphKnowsClient, InProcessGraphKnows
from services.agent.tools.search.client import SearchServiceClient
from services.agent.tools.search.web import TavilySearchTool
from services.agent.tools.search.cellar import CellarSPARQLTool


def _configure_logging() -> None:
    """Configure root logger for structured JSON output."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


_configure_logging()
log = logging.getLogger(__name__)

_settings: AgentSettings | None = None
_sgm: SessionGraphManager | None = None
_http_client: httpx.AsyncClient | None = None
_graph_backend = None
_gk_container = None  # GraphKnowsContainer (inprocess mode only)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _sgm, _http_client, _graph_backend, _gk_container
    _settings = AgentSettings()
    _settings.check_production_secrets()
    _http_client = httpx.AsyncClient(timeout=_settings.httpx_timeout)
    _sgm = SessionGraphManager(
        memgraph_uri=_settings.memgraph_uri,
        neo4j_uri=_settings.neo4j_uri,
        neo4j_user=_settings.neo4j_user,
        neo4j_password=_settings.neo4j_password,
    )

    if SESSION_GRAPH_BACKEND == "neo4j":
        _graph_backend = Neo4jGDSBackend(
            _settings.neo4j_uri,
            _settings.neo4j_user,
            _settings.neo4j_password,
        )
    else:
        _graph_backend = MemgraphBackend(_settings.memgraph_uri)
    await _graph_backend.ensure_indexes()

    # --- graphknows client: http (default) or inprocess ---
    if _settings.graphknows_mode == "inprocess":
        from graphknows import GraphKnows
        from graphknows.config import GraphKnowsSettings
        gk_cfg = GraphKnowsSettings(
            neo4j_uri=_settings.neo4j_uri,
            neo4j_user=_settings.neo4j_user,
            neo4j_password=_settings.neo4j_password,
            pgvector_dsn=_settings.pgvector_dsn,
            memgraph_uri=_settings.memgraph_uri,
        )
        _gk_container = await GraphKnows.from_settings(gk_cfg).__aenter__()
        graphknows_client = InProcessGraphKnows(_gk_container)
        log.info("graphknows mode: inprocess")
    else:
        graphknows_client = GraphKnowsClient(base_url=_settings.graphknows_url)
        log.info("graphknows mode: http (%s)", _settings.graphknows_url)

    app.state.graph_backend = _graph_backend
    search_client = SearchServiceClient(base_url=_settings.search_url)
    tavily = TavilySearchTool() if os.environ.get("TAVILY_API") or os.environ.get("TAVILY_API_KEY") else None
    cellar = CellarSPARQLTool()
    from services.agent.tools.extraction.in_process import InProcessExtractor as load_extractor
    extractor = load_extractor()
    _init_router(_settings, _sgm, _http_client, search_client, _graph_backend, tavily=tavily, cellar=cellar, extractor=extractor, graphknows_client=graphknows_client)
    app.state.settings = _settings
    app.state.sgm = _sgm
    app.state.http_client = _http_client
    app.state.search_client = search_client
    yield
    await _http_client.aclose()
    if _gk_container is not None:
        await _gk_container.__aexit__(None, None, None)


app = FastAPI(title="Graphknows Agent", version="1.0.0", lifespan=lifespan)
app.include_router(router)
