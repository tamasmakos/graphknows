"""The symbol writer: the concept and predicate index vertices."""

from __future__ import annotations

from collections.abc import Sequence

from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.storage.arcadedb._sql import vector_literal
from graphknows.storage.arcadedb.writers._base import _Writer


class SymbolWriter(_Writer):
    """Writes CONCEPT and PREDICATE vertices — the symbolic index layer."""

    async def write_concept(self, concept: ConceptRef, embedding: Sequence[float]) -> str:
        """MERGE the CONCEPT vertex and return its uri.

        The embedding of the definition is written with the symbol, not beside
        it: it is what makes the concept retrievable at all. Inlined, because
        a LIST parameter is rejected on a vector-indexed property; an empty one
        is left unset.
        """
        embedding_sql = f", c.embedding = {vector_literal(embedding)}" if embedding else ""
        await self._store.command(
            "MERGE (c:CONCEPT {uri: $uri}) "
            f"SET c.label = $label, c.definition = $definition, c.pack = $pack{embedding_sql}",
            uri=concept.uri,
            label=concept.label,
            definition=concept.definition,
            pack=concept.pack,
        )
        return concept.uri

    async def write_evokes(self, segment_id: str, concept_uri: str, confidence: float) -> None:
        """MERGE the SEGMENT-[EVOKES]->CONCEPT edge of bottom-up labelling (FR-002)."""
        await self._store.command(
            "MATCH (g:SEGMENT {id: $segment_id}), (c:CONCEPT {uri: $uri}) "
            "MERGE (g)-[e:EVOKES]->(c) SET e.confidence = $confidence",
            segment_id=segment_id,
            uri=concept_uri,
            confidence=confidence,
        )

    async def write_instance_of(self, entity_id: str, concept: ConceptRef, score: float) -> None:
        """MERGE the ENTITY-[INSTANCE_OF]->CONCEPT edge typing one entity (FR-002).

        The owning pack is carried on the edge: two packs may type the same
        entity, and a reader must be able to tell whose judgement it is.
        """
        await self._store.command(
            "MATCH (e:ENTITY {id: $entity_id}), (c:CONCEPT {uri: $uri}) "
            "MERGE (e)-[i:INSTANCE_OF]->(c) SET i.score = $score, i.pack = $pack",
            entity_id=entity_id,
            uri=concept.uri,
            score=score,
            pack=concept.pack,
        )

    async def write_predicate(self, predicate: PredicateRef) -> str:
        """MERGE the PREDICATE vertex and return its id."""
        await self._store.command(
            "MERGE (p:PREDICATE {id: $id}) "
            "SET p.label = $label, p.definition = $definition, p.canonical = $canonical, "
            "p.functional = $functional, p.domain = $domain, p.range = $range, p.pack = $pack",
            id=predicate.id,
            label=predicate.label,
            definition=predicate.definition,
            canonical=predicate.canonical,
            functional=predicate.functional,
            domain=predicate.domain or "",
            range=predicate.range or "",
            pack=predicate.pack,
        )
        return predicate.id
