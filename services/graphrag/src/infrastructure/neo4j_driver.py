"""
Async Neo4j driver for the graphrag service.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase


def _settings():
    from src.common.config.settings import AppSettings

    return AppSettings()


def create_driver() -> AsyncDriver:
    cfg = _settings()
    return AsyncGraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_user, cfg.neo4j_password),
    )


@asynccontextmanager
async def get_driver() -> AsyncIterator[AsyncDriver]:
    driver = create_driver()
    try:
        await driver.verify_connectivity()
        yield driver
    finally:
        await driver.close()
