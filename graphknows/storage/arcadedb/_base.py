"""Shared lifecycle for ArcadeDB stores that own exactly one database.

``GraphStore``, ``ArcadeDBLTMStore``, and ``ArcadeDBSTMStore`` each repeated the
same block verbatim: the client/database accessors, connect/close/drop, the
query/command pair, and a DDL loop that treats "already exists" as success. The
vector store deliberately does not inherit from this — it spans two databases,
so the single-``_db`` contract here would not fit it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

# Placeholder swapped for the real embedding dimension at schema-creation time,
# so vector-index DDL can be declared as a module-level constant.
VEC_DIM_TOKEN = "__VEC_DIM__"


class ArcadeStoreBase:
    """Connection handling and idempotent DDL for one ArcadeDB database."""

    def __init__(self, client: ArcadeDBClient, db: str) -> None:
        self._c = client
        self._db = db

    @property
    def client(self) -> ArcadeDBClient:
        """The underlying ArcadeDB client (may be shared across stores)."""
        return self._c

    @property
    def database(self) -> str:
        """Name of the ArcadeDB database this store reads/writes."""
        return self._db

    async def connect(self) -> None:
        """Open the shared client connection. Idempotent.

        Schema creation is deliberately not run here — call ``ensure_schema()``
        explicitly on write paths so read paths avoid idempotent DDL round-trips.
        """
        await self._c.connect()

    async def close(self) -> None:
        """Close the shared client connection. Idempotent."""
        await self._c.close()

    async def drop_database(self) -> None:
        """Drop this store's entire ArcadeDB database (destructive)."""
        await self._c.drop_database(self._db)

    async def query(self, cypher: str, **params: Any) -> list[dict]:
        """Run a read-only statement against this store's database."""
        return await self._c.query(self._db, cypher, params=params or None)

    async def command(self, cypher: str, **params: Any) -> list[dict]:
        """Run a writing statement against this store's database.

        Public because channels, ingestion steps, and the web server legitimately
        need raw Cypher these classes do not wrap; they used to reach for a
        private method, which made renaming it a silent break across five modules.
        """
        return await self._c.command(self._db, cypher, params=params or None)

    async def _apply_ddl(
        self,
        statements: Iterable[str | tuple[str, str]],
        *,
        dims: int | None = None,
    ) -> None:
        """Create the database, then execute schema DDL idempotently.

        Accepts bare SQL strings or ``(language, statement)`` pairs. Re-creating
        existing types is expected and ignored; any other failure is re-raised,
        so a genuine DDL error is never mistaken for benign re-creation.

        Args:
            statements: DDL to execute in order.
            dims: Embedding dimension substituted for :data:`VEC_DIM_TOKEN`.
        """
        await self._c.create_database(self._db)
        name = type(self).__name__
        for entry in statements:
            lang, stmt = ("sql", entry) if isinstance(entry, str) else entry
            if dims is not None:
                stmt = stmt.replace(VEC_DIM_TOKEN, str(dims))
            try:
                await self._c.command(self._db, stmt, language=lang)
            except Exception as exc:
                exc_s = str(exc).lower()
                if "already exists" in exc_s or "duplicate" in exc_s:
                    logger.debug("%s DDL already exists (idempotent): %s", name, stmt[:60])
                    continue
                logger.warning("%s DDL failed: %s | %s", name, stmt[:60], exc)
                raise
