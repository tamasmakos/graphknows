"""Graphknows Agent Service entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from agent.config import AgentSettings
from agent.infra.pgvector_client import PgVectorClient
from agent.session.manager import SessionGraphManager
from agent.router import _init_router, router
from services.agent.config import SESSION_GRAPH_BACKEND
from services.agent.graph.backends.memgraph import MemgraphBackend
from services.agent.graph.backends.neo4j_gds import Neo4jGDSBackend
from services.agent.tools.search_client import SearchServiceClient


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _sgm, _http_client, _graph_backend
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

    pg_client = PgVectorClient(dsn=_settings.pgvector_dsn)
    await pg_client.connect()
    await pg_client.ensure_schema()
    app.state.pg_client = pg_client
    app.state.graph_backend = _graph_backend
    search_client = SearchServiceClient(base_url=_settings.search_url)
    _init_router(_settings, _sgm, _http_client, pg_client, search_client, _graph_backend)
    app.state.settings = _settings
    app.state.sgm = _sgm
    app.state.http_client = _http_client
    app.state.search_client = search_client
    yield
    await _http_client.aclose()
    await app.state.pg_client.close()


app = FastAPI(title="Graphknows Agent", version="1.0.0", lifespan=lifespan)
app.include_router(router)
