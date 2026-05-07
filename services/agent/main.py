"""Graphknows Agent Service entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from agent.config import AgentSettings
from agent.session.manager import SessionGraphManager
from agent.router import _init_router, router


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _sgm, _http_client
    _settings = AgentSettings()
    _settings.check_production_secrets()
    _http_client = httpx.AsyncClient(timeout=_settings.httpx_timeout)
    _sgm = SessionGraphManager(
        memgraph_uri=_settings.memgraph_uri,
        neo4j_uri=_settings.neo4j_uri,
        neo4j_user=_settings.neo4j_user,
        neo4j_password=_settings.neo4j_password,
    )
    _init_router(_settings, _sgm, _http_client)
    app.state.settings = _settings
    app.state.sgm = _sgm
    app.state.http_client = _http_client
    yield
    await _http_client.aclose()


app = FastAPI(title="Graphknows Agent", version="1.0.0", lifespan=lifespan)
app.include_router(router)
