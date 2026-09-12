"""The ENTITY writer: resolved entities and the mentions that evidence them."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from graphknows.models.fact import Mention
from graphknows.storage.arcadedb._sql import vector_literal
from graphknows.storage.arcadedb.writers._base import _Writer


@dataclass(frozen=True)
class EntityWrite:
    """One entity as identity resolution resolved it.

    Nothing here is derived by the writer: ``name_norm`` and ``block_key`` are
    the resolver's decisions, and a writer that recomputed them would be a
    second, silently diverging normaliser. ``type_histogram`` is what this
    write adds — the labels the entity was read under this time — and the
    writer folds it into the tally already stored.
    """

    id: str
    name: str
    name_norm: str
    block_key: str = ""
    embedding: list[float] = field(default_factory=list)
    type_histogram: dict[str, int] = field(default_factory=dict)


class EntityWriter(_Writer):
    """Writes ENTITY vertices and MENTIONS edges from their evidence segments."""

    async def write(self, entity: EntityWrite) -> str:
        """UPSERT the ENTITY vertex and return its id.

        SQL rather than Cypher: ``type_histogram`` is a MAP, which the Cypher
        dialect cannot bind, and the embedding is inlined because a LIST
        parameter is rejected on a vector-indexed property. An empty embedding
        is left unset — the index refuses a zero-length vector.
        """
        histogram = Counter(await self._histogram(entity.id))
        histogram.update(entity.type_histogram)
        embedding = f", embedding = {vector_literal(entity.embedding)}" if entity.embedding else ""
        await self._store.sql_command(
            # The only interpolation is a float vector literal.
            "UPDATE ENTITY SET name = :name, name_norm = :name_norm, "  # nosec B608
            "block_key = :block_key, "
            "type_histogram = :type_histogram, state = coalesce(state, 'active')"
            f"{embedding} UPSERT WHERE id = :id",
            id=entity.id,
            name=entity.name,
            name_norm=entity.name_norm,
            block_key=entity.block_key,
            type_histogram=dict(histogram),
        )
        return entity.id

    async def _histogram(self, entity_id: str) -> dict[str, int]:
        """The label tally already stored on the entity; empty when it is new."""
        rows = await self._store.query(
            "MATCH (e:ENTITY {id: $id}) RETURN e.type_histogram AS histogram",
            id=entity_id,
        )
        return dict(rows[0].get("histogram") or {}) if rows else {}

    async def write_mention(self, mention: Mention) -> None:
        """MERGE the SEGMENT-[MENTIONS]->ENTITY edge for one surface form.

        The span is kept on the edge so a recalled entity can quote the words
        it was read from, rather than only the segment they sit in.
        """
        start, end = mention.span
        await self._store.command(
            "MATCH (g:SEGMENT {id: $segment_id}), (e:ENTITY {id: $entity_id}) "
            "MERGE (g)-[m:MENTIONS {surface: $surface, span: $span}]->(e) "
            "SET m.confidence = $confidence, m.extractor = $extractor",
            segment_id=mention.segment_id,
            entity_id=mention.entity_id,
            surface=mention.surface,
            span=f"{start}:{end}",
            confidence=mention.confidence,
            extractor=mention.extractor,
        )
