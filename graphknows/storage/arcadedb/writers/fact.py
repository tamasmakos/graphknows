"""The FACT writer: assertions, and the evidence without which none is written."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from graphknows.exceptions import StoreError
from graphknows.models.fact import Fact
from graphknows.storage.arcadedb.writers._base import _Writer


@dataclass(frozen=True)
class Evidence:
    """The segment a fact was asserted in, and where in it (FR-008).

    Attributes:
        segment_id: The evidence segment.
        span: ``(start, end)`` byte offsets of the assertion within it.
    """

    segment_id: str
    span: tuple[int, int]


class FactWriter(_Writer):
    """Writes FACT vertices with their subject, object, predicate and evidence."""

    async def write(self, fact: Fact, evidence: Sequence[Evidence]) -> str:
        """Write one fact and its edges in a single transaction; return its id.

        A fact with no evidence is rejected here rather than written and
        repaired later: unsourced facts are indistinguishable from sourced ones
        once they are in the graph, so the only place the invariant holds is
        the write itself (FR-008).
        """
        if not evidence:
            raise StoreError(f"fact {fact.id!r} has no ASSERTED_IN evidence")
        async with self._store.transaction():
            await self._write_vertex(fact)
            await self._link_triple(fact)
            for item in evidence:
                await self._link_evidence(fact.id, item)
        return fact.id

    async def _write_vertex(self, fact: Fact) -> None:
        """MERGE the FACT vertex itself."""
        await self._store.command(
            "MERGE (f:FACT {id: $id}) "
            "SET f.subject = $subject, f.predicate = $predicate, f.object = $object, "
            "f.polarity = $polarity, f.modality = $modality, f.confidence = $confidence, "
            "f.valid_from = $valid_from, f.valid_to = $valid_to, f.is_current = $is_current, "
            "f.extractor = $extractor, f.extractor_version = $extractor_version, "
            "f.state = $state",
            id=fact.id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            polarity=str(fact.polarity),
            modality=fact.modality or "",
            confidence=fact.confidence,
            valid_from=_iso(fact.validity.valid_from),
            valid_to=_iso(fact.validity.valid_to),
            is_current=fact.is_current,
            extractor=fact.extractor,
            extractor_version=fact.extractor_version,
            state=str(fact.state),
        )

    async def _link_triple(self, fact: Fact) -> None:
        """MERGE the subject, object and predicate edges of the assertion."""
        await self._store.command(
            "MATCH (f:FACT {id: $id}), (s:ENTITY {id: $subject}), "
            "(o:ENTITY {id: $object}), (p:PREDICATE {id: $predicate}) "
            "MERGE (s)-[:SUBJECT_OF]->(f) "
            "MERGE (f)-[:OBJECT_IS]->(o) "
            "MERGE (f)-[:USES]->(p)",
            id=fact.id,
            subject=fact.subject,
            object=fact.object,
            predicate=fact.predicate,
        )

    async def _link_evidence(self, fact_id: str, evidence: Evidence) -> None:
        """MERGE one FACT-[ASSERTED_IN]->SEGMENT edge, carrying its byte span."""
        start, end = evidence.span
        await self._store.command(
            "MATCH (f:FACT {id: $id}), (g:SEGMENT {id: $segment_id}) "
            "MERGE (f)-[a:ASSERTED_IN]->(g) SET a.span = $span",
            id=fact_id,
            segment_id=evidence.segment_id,
            span=f"{start}:{end}",
        )


def _iso(moment: datetime | None) -> str:
    """Render an optional validity end for storage; an open end stays empty."""
    return moment.isoformat() if moment else ""
