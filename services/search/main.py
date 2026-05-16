"""Graphknows Search Service — FastAPI application entry point.

Set SEARCH_MODE=mcp to run as an MCP stdio server instead of HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .connector_registry import get_registry
from .router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = get_registry()
    logger.info(
        "Graphknows Search Service starting — connectors: %s",
        registry.list_ids(),
    )
    yield
    logger.info("Graphknows Search Service shutting down")


app = FastAPI(
    title="Graphknows EU Search Service",
    description=(
        "Unified search across EU parliamentary and public data sources. "
        "Supports EP debates, EUR-Lex, Council conclusions, ECB, EPRS, and more."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


def main() -> None:
    """Entry point: HTTP (default) or MCP stdio based on SEARCH_MODE env var."""
    if os.getenv("SEARCH_MODE", "").lower() == "mcp":
        from .mcp_server import mcp_main
        asyncio.run(mcp_main())
    else:
        import uvicorn
        uvicorn.run(
            "services.search.main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            log_level="info",
        )


if __name__ == "__main__":
    main()
