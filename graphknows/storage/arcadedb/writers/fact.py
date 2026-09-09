"""The FACT writer: assertions, and the evidence without which none is written."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from graphknows.exceptions import StoreError
from graphknows.models.fact import Fact, FactState
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

    async def incumbent(self, fact: Fact) -> tuple[str, datetime | None] | None:
        """The other current fact on *fact*'s subject and predicate, with its anchor.

        ``None`` when nothing yet claims that subject and predicate — asked only
        for functional predicates, where at most one object may be current.
        """
        rows = await self._store.query(
            "MATCH (f:FACT) WHERE f.subject = $subject AND f.predicate = $predicate "
            "AND f.is_current = true AND f.state = $state AND f.id <> $id "
            "RETURN f.id AS id, f.valid_from AS valid_from LIMIT 1",
            subject=fact.subject,
            predicate=fact.predicate,
            state=str(FactState.ACTIVE),
            id=fact.id,
        )
        if not rows:
            return None
        return str(rows[0]["id"]), moment(rows[0].get("valid_from"))

    async def supersede(self, winner_id: str, loser_id: str) -> None:
        """Link the newer fact to the older one it retires (FR-012).

        The loser is kept and marked not current: superseding is a tombstone,
        never a delete, so the history of a functional predicate stays readable.
        """
        await self._store.command(
            "MATCH (w:FACT {id: $winner}), (l:FACT {id: $loser}) "
            "MERGE (w)-[:SUPERSEDES]->(l) SET l.is_current = false",
            winner=winner_id,
            loser=loser_id,
        )

    async def contradict(self, fact_id: str, other_id: str) -> None:
        """Link two facts claiming one functional predicate at the same instant.

        Neither can be called stale by time alone, so both stay current and the
        conflict is recorded rather than resolved.
        """
        await self._store.command(
            "MATCH (f:FACT {id: $id}), (o:FACT {id: $other}) MERGE (f)-[:CONTRADICTS]->(o)",
            id=fact_id,
            other=other_id,
        )

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


def _iso(end: datetime | None) -> str:
    """Render an optional validity end for storage; an open end stays empty."""
    return end.isoformat() if end else ""


def moment(stored: object) -> datetime | None:
    """Read back what :func:`_iso` wrote; an empty or absent end stays open."""
    return datetime.fromisoformat(str(stored)) if stored else None
