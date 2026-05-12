"""Graphknows Search Service — FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .router import router, _ADAPTERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: log registered adapters
    logger.info(
        "Graphknows Search Service starting — adapters: %s",
        list(_ADAPTERS.keys()),
    )
    yield
    # Shutdown: nothing to clean up for in-memory adapters
    logger.info("Graphknows Search Service shutting down")


app = FastAPI(
    title="Graphknows Search Service",
    description=(
        "FastAPI service wrapping EP Debates, EUR-Lex, and UK Legislation scrapers. "
        "Provides unified search and async scrape-job endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")
