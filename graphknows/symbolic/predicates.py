"""The predicate index — the second symbol kind, resolved from what a decoder said.

An extractor answers with a relation *string*; a fact names a predicate *symbol*
(FR-011). This index is the one place that crossing happens: it holds a pack's
:class:`~graphknows.models.symbols.PredicateRef` vocabulary and answers with the
symbol a surface form names, so nothing downstream ever holds a free string.

Every surface a predicate is reachable by — its label, its canonical rendering
and its id — is normalised the same way, so ``works_at``, ``Works At`` and
``works at`` all reach one symbol.
"""

from __future__ import annotations

from collections.abc import Iterable

from graphknows.models.symbols import PredicateRef


def _surface(text: str) -> str:
    """Comparison form of a predicate surface: case, spacing and underscores folded."""
    return " ".join(text.replace("_", " ").split()).casefold()


class PredicateIndex:
    """A pack's predicates, keyed by every surface form they answer to.

    First declaration of a surface wins: a later predicate never silently steals
    the label an earlier one already answers to.
    """

    def __init__(self, predicates: Iterable[PredicateRef]) -> None:
        self._by_surface: dict[str, PredicateRef] = {}
        for predicate in predicates:
            for text in (predicate.label, predicate.canonical, predicate.id):
                self._by_surface.setdefault(_surface(text), predicate)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} surfaces)"

    def __len__(self) -> int:
        return len(self._by_surface)

    # ponytail: normalised surface match only; add definition-embedding cosine
    # (graphknows.symbolic.index) if decoder phrasing drifts off pack labels.
    def resolve(self, label: str) -> PredicateRef | None:
        """The predicate *label* names, or ``None`` when the pack names no such relation."""
        return self._by_surface.get(_surface(label))


__all__ = ["PredicateIndex"]
