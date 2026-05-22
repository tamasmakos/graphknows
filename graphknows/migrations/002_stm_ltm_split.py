"""Migration 002: STM/LTM split — drop legacy tables, create new schema.

Drops: chunks, entities
Creates: stm_chunks, stm_turns, ltm_chunks, ltm_turns, ltm_topics
"""

from __future__ import annotations

import asyncio
import logging
import os

from graphknows.retrieval.stores.pgvector import PgVectorStore

log = logging.getLogger(__name__)

_DROP_LEGACY = """
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
"""


async def run(dsn: str) -> None:
    """Execute the migration against *dsn*.

    Args:
        dsn: PostgreSQL connection string.
    """
    store = PgVectorStore(dsn)
    await store.connect()
    try:
        pool = store._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(_DROP_LEGACY)
            log.info("Dropped legacy tables: chunks, entities")
        await store.ensure_schema()
        log.info("Migration 002 complete: STM/LTM tables created")
    finally:
        await store.close()


if __name__ == "__main__":
    dsn = os.environ["PGVECTOR_DSN"]
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(dsn))
