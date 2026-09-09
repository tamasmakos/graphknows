"""What a domain pack must offer the core — domain knowledge as data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, runtime_checkable

from graphknows.models import ConceptRef, PredicateRef


@runtime_checkable
class DomainPack(Protocol):
    """A unit of domain knowledge the core loads but never imports.

    ``lexicon()``, ``channel()``, ``expand()`` and ``retrieval_profile()`` arrive
    with the dialogue pack, where each gets its first caller (FR-022); so do
    ``parser()`` and ``extractor()``, whose value types this slice has not created
    yet.
    """

    @property
    def name(self) -> str:
        """Pack name, reported when two packs claim the same key."""

    def concepts(self) -> Iterable[ConceptRef]:
        """The pack's concepts, iterated lazily — never materialised eagerly."""

    def predicates(self) -> Iterable[PredicateRef]:
        """The pack's predicates, iterated lazily."""

    def entity_labels(self) -> tuple[str, ...]:
        """The extraction label set — it *replaces*, it never unions (FR-022)."""

    def prompt_addendum(self) -> str:
        """Domain guidance appended to the extraction prompt."""

    def hygiene(self) -> Callable[[str], bool]:
        """The pack's name hygiene: True when a surface may become an entity (FR-022)."""

    def thresholds(self) -> Mapping[str, float]:
        """Pack-supplied thresholds, by name: ``mention`` and ``fact`` confidence floors."""

    def veto(self, a: str, b: str) -> bool:
        """True when the pack forbids merging these two entity ids (FR-019)."""
