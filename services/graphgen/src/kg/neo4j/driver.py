"""
Async Neo4j driver factory.

Usage
-----
    async with get_driver() as driver:
        async with driver.session(database="neo4j") as session:
            result = await session.run("MATCH (n) RETURN count(n) AS c")
            record = await result.single()
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase


def _uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


def _auth() -> tuple[str, str]:
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "changeme")
    return user, password


def create_driver() -> AsyncDriver:
    """Create a new AsyncDriver instance (caller is responsible for closing)."""
    return AsyncGraphDatabase.driver(_uri(), auth=_auth())


@asynccontextmanager
async def get_driver() -> AsyncIterator[AsyncDriver]:
    """Context manager that opens and closes an async Neo4j driver."""
    driver = create_driver()
    try:
        await driver.verify_connectivity()
        yield driver
    finally:
        await driver.close()
