"""The ENTITY writer: resolved entities and the mentions that evidence them."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphknows.models.fact import Mention
from graphknows.storage.arcadedb.writers._base import _Writer


@dataclass(frozen=True)
class EntityWrite:
    """One entity as identity resolution resolved it.

    Nothing here is derived by the writer: ``name_norm`` and ``block_key`` are
    the resolver's decisions, and a writer that recomputed them would be a
    second, silently diverging normaliser.
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
        """MERGE the ENTITY vertex and return its id."""
        await self._store.command(
            "MERGE (e:ENTITY {id: $id}) "
            "SET e.name = $name, e.name_norm = $name_norm, e.block_key = $block_key, "
            "e.embedding = $embedding, e.type_histogram = $type_histogram, "
            "e.state = coalesce(e.state, 'active')",
            id=entity.id,
            name=entity.name,
            name_norm=entity.name_norm,
            block_key=entity.block_key,
            embedding=entity.embedding,
            type_histogram=entity.type_histogram,
        )
        return entity.id

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
