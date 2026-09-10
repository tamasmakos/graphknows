"""The code pack: source-code vocabulary as data, plus its identity rules."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from graphknows.models import ConceptRef, PredicateRef

PACK_NAME = "code"

_DATA = Path(__file__).resolve().parent.parent / "data" / "code.json"

_SYMBOL_PREFIX = "code"


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    """The pack's vocabulary file, read once per process."""
    data: dict[str, Any] = json.loads(_DATA.read_text(encoding="utf-8"))
    return data


def symbol_id(kind: str, name: str) -> str:
    """Identity of a code symbol, qualified by *kind*.

    A code name means nothing without its kind: a package and its entry-point
    function share a name all the time. Carrying the kind in the id is what
    gives :meth:`CodePack.veto` something to read (FR-019).
    """
    return f"{_SYMBOL_PREFIX}:{kind}:{' '.join(name.split()).casefold()}"


def _parts(entity_id: str) -> tuple[str, str] | None:
    """*entity_id* as ``(kind, name)``, or ``None`` when this pack did not mint it."""
    prefix, _, rest = entity_id.partition(":")
    kind, sep, name = rest.partition(":")
    return (kind, name) if prefix == _SYMBOL_PREFIX and sep else None


class CodePack:
    """Domain knowledge about source code, loaded by the core and never imported by it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @property
    def name(self) -> str:
        """Pack name, reported when two packs claim the same key."""
        return PACK_NAME

    def concepts(self) -> Iterable[ConceptRef]:
        """Function, class, module and file (FR-029), read lazily from the data file."""
        for concept in _data()["concepts"]:
            yield ConceptRef(**concept, pack=PACK_NAME)

    def predicates(self) -> Iterable[PredicateRef]:
        """Calls, imports, defines and tests (FR-029), read lazily from the data file."""
        for predicate in _data()["predicates"]:
            yield PredicateRef(**predicate, pack=PACK_NAME)

    def entity_labels(self) -> tuple[str, ...]:
        """The concept labels, which are the only labels a code mention may carry."""
        return tuple(concept["label"] for concept in _data()["concepts"])

    def extractor(self) -> Any:
        """This pack reads its own domain: the syntax tree, not a model (FR-030).

        Imported here rather than at module scope because the extractor mints
        its ids with :func:`symbol_id` from this module.
        """
        from graphknows.packs.code.extractor import CodeExtractor

        return CodeExtractor()

    def prompt_addendum(self) -> str:
        """Guidance for the assisted path; the deterministic one never reads it."""
        return (
            "Name code symbols exactly as written in the source: a bare function "
            "or class name, or a dotted module path. Never invent a symbol the "
            "text does not contain."
        )

    def hygiene(self) -> Callable[[str], bool]:
        """Admits identifiers and dotted paths; prose is not a code symbol (FR-022)."""
        return _is_symbol

    def thresholds(self) -> Mapping[str, float]:
        """Syntax is certain: a code mention or fact below full confidence is a guess."""
        return {"mention": 1.0, "fact": 1.0}

    def veto(self, a: str, b: str) -> bool:
        """True when *a* and *b* are a function and a module of one name (FR-019).

        ``run`` the module and ``run`` the function share every weak signal there
        is — the same normalised name, the same repository, neighbouring text —
        so the veto is what keeps them two entities.
        """
        first, second = _parts(a), _parts(b)
        if first is None or second is None:
            return False
        return first[1] == second[1] and {first[0], second[0]} == {"function", "module"}


def _is_symbol(surface: str) -> bool:
    """True when *surface* could be written as a code symbol."""
    parts = surface.split(".")
    return bool(parts) and all(part.isidentifier() for part in parts)
