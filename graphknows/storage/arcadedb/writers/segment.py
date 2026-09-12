"""The SEGMENT writer: evidence spans, each attached to its source."""

from __future__ import annotations

from collections.abc import Sequence

from graphknows.models.segment import Segment
from graphknows.storage.arcadedb._sql import vector_literal
from graphknows.storage.arcadedb.writers._base import _Writer


class SegmentWriter(_Writer):
    """Writes SEGMENT vertices, their PART_OF edge to the source, and NEXT adjacency."""

    async def write(self, segment: Segment, embedding: Sequence[float] = ()) -> str:
        """MERGE the SEGMENT vertex, link it to its source, return its id.

        The PART_OF edge is written here rather than by a caller: a segment
        that is not part of a source is evidence nothing can cite. The
        embedding is inlined — a LIST parameter is rejected on a
        vector-indexed property — and an empty one is left unset.
        """
        start, end = segment.byte_range
        observed_at = segment.observed_at.isoformat() if segment.observed_at else ""
        embedding_sql = f", g.embedding = {vector_literal(embedding)}" if embedding else ""
        await self._store.command(
            "MERGE (g:SEGMENT {id: $id}) "
            "SET g.source_id = $source_id, g.text = $text, g.kind = $kind, g.path = $path, "
            "g.byte_start = $byte_start, g.byte_end = $byte_end, g.role = $role, "
            "g.observed_at = $observed_at, g.observed_at_inferred = $observed_at_inferred, "
            f"g.extractor_version = $extractor_version{embedding_sql}",
            id=segment.id,
            source_id=segment.source_id,
            text=segment.text,
            kind=str(segment.kind),
            path=segment.path,
            byte_start=start,
            byte_end=end,
            role=segment.role,
            observed_at=observed_at,
            observed_at_inferred=segment.observed_at_inferred,
            extractor_version=segment.extractor_version,
        )
        await self._store.command(
            "MATCH (g:SEGMENT {id: $id}), (s:SOURCE {id: $source_id}) MERGE (g)-[:PART_OF]->(s)",
            id=segment.id,
            source_id=segment.source_id,
        )
        return segment.id

    async def link_next(self, previous_id: str, segment_id: str) -> None:
        """MERGE the SEGMENT-[NEXT]->SEGMENT edge between two consecutive segments.

        Written by the pipeline in source order, so a neighbour walk can reach
        the sibling a long message was split into.
        """
        await self._store.command(
            "MATCH (a:SEGMENT {id: $previous_id}), (b:SEGMENT {id: $segment_id}) "
            "MERGE (a)-[:NEXT]->(b)",
            previous_id=previous_id,
            segment_id=segment_id,
        )
