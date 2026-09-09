"""Load packs all-or-nothing: one contested key and none of them is loaded."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from graphknows.exceptions import PackConflictError
from graphknows.models import ConceptRef, PredicateRef
from graphknows.packs.protocol import DomainPack

_Symbol = TypeVar("_Symbol", ConceptRef, PredicateRef)


@dataclass(frozen=True, slots=True)
class LoadedPacks:
    """The symbol tables of every loaded pack, keyed globally.

    Attributes:
        concepts: Concepts by uri, across all packs.
        predicates: Predicates by id, across all packs.
    """

    concepts: dict[str, ConceptRef]
    predicates: dict[str, PredicateRef]


def _index(claims: Iterable[tuple[str, _Symbol]]) -> dict[str, _Symbol]:
    """Table the claimed symbols, refusing a key two packs both claim."""
    table: dict[str, _Symbol] = {}
    for key, symbol in claims:
        if (owner := table.get(key)) is not None:
            raise PackConflictError(key, owner.pack, symbol.pack)
        table[key] = symbol
    return table


def load_packs(packs: Sequence[DomainPack]) -> LoadedPacks:
    """Merge the packs' symbols into one namespace (FR-004, FR-025).

    Raises:
        PackConflictError: Two packs claim one concept uri or predicate id.
            Nothing is returned, so no pack is left loaded.
    """
    return LoadedPacks(
        concepts=_index((concept.uri, concept) for pack in packs for concept in pack.concepts()),
        predicates=_index(
            (predicate.id, predicate) for pack in packs for predicate in pack.predicates()
        ),
    )
