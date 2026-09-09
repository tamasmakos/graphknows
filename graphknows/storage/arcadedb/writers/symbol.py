"""The symbol writer: the concept and predicate index vertices."""

from __future__ import annotations

from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.storage.arcadedb.writers._base import _Writer


class SymbolWriter(_Writer):
    """Writes CONCEPT and PREDICATE vertices — the symbolic index layer."""

    async def write_concept(self, concept: ConceptRef, embedding: list[float]) -> str:
        """MERGE the CONCEPT vertex and return its uri.

        The embedding of the definition is written with the symbol, not beside
        it: it is what makes the concept retrievable at all.
        """
        await self._store.command(
            "MERGE (c:CONCEPT {uri: $uri}) "
            "SET c.label = $label, c.definition = $definition, "
            "c.embedding = $embedding, c.pack = $pack",
            uri=concept.uri,
            label=concept.label,
            definition=concept.definition,
            embedding=embedding,
            pack=concept.pack,
        )
        return concept.uri

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
