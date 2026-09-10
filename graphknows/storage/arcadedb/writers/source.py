"""The SOURCE writer: one ingested artefact."""

from __future__ import annotations

from graphknows.models.source import Source
from graphknows.storage.arcadedb.writers._base import _Writer


class SourceWriter(_Writer):
    """Writes SOURCE vertices."""

    async def write(self, source: Source) -> str:
        """UPSERT the SOURCE vertex and return its id.

        Keyed on the source's derived id, so re-importing the same bytes from
        the same uri updates one vertex rather than forking a second. SQL
        rather than Cypher: ``meta`` is a MAP, which the Cypher dialect cannot
        bind as a parameter.
        """
        await self._store.sql_command(
            "UPDATE SOURCE SET uri = :uri, mime = :mime, content_hash = :content_hash, "
            "imported_at = :imported_at, namespace = :namespace, meta = :meta "
            "UPSERT WHERE id = :id",
            id=source.id,
            uri=source.uri,
            mime=source.mime,
            content_hash=source.content_hash,
            imported_at=source.imported_at.isoformat(),
            namespace=source.namespace,
            meta=source.meta,
        )
        return source.id

    async def exists(self, content_hash: str) -> bool:
        """Whether these bytes are already stored, under any uri (FR-007)."""
        rows = await self._store.query(
            "MATCH (s:SOURCE {content_hash: $content_hash}) RETURN s.id AS id LIMIT 1",
            content_hash=content_hash,
        )
        return bool(rows)
