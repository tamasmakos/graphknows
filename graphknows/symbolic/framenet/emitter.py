"""FrameNet as a concept+predicate emitter — the only shape frames take now.

A frame is a :class:`~graphknows.models.symbols.ConceptRef` and each of its core
frame elements is a :class:`~graphknows.models.symbols.PredicateRef` whose domain
is that concept (FR-023). No frame symbol kind, no frame-element mirror, no
frame-specific vertex or edge type: a pack hands these two tuples to the loader
and the symbol index carries them like any other pack's vocabulary.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from graphknows.models.symbols import ConceptRef, PredicateRef

_MARKUP = re.compile(r"<[^>]+>")

# What the corpus calls a role that the frame itself is about. Peripheral and
# Extra-Thematic elements (time, manner, place) are not the frame's own
# relations, so they earn no predicate.
_CORE = "Core"


def _clean(text: str | None) -> str:
    """Definition text without FrameNet's inline markup, capped for embedding."""
    return _MARKUP.sub("", text or "").strip()[:400]


def _framenet() -> Any:
    """The FrameNet reader.

    ``nltk.corpus.framenet`` is a ``LazyCorpusLoader``: importing it never touches
    disk, so the corpus is only resolved on first *use* (e.g. ``fn.frames()``). The
    guard has to sit around that use, not the import above it.

    A missing corpus is an ENVIRONMENT defect, so this neither downloads it nor
    passes over it quietly. It does not download because this is library code: a
    package installed from PyPI must not reach the network mid-ingest, and
    ``nltk.download()`` without an explicit ``download_dir`` ignores ``NLTK_DATA``
    anyway (see ``scripts/bake_models.py``), so the fetch would land somewhere
    nothing reads.
    """
    from nltk.corpus import framenet as fn

    try:
        fn.frames()  # probe: raises LookupError if the corpus isn't downloaded
    except LookupError as exc:
        raise RuntimeError(
            "FrameNet corpus 'framenet_v17' not found. Provision it rather than "
            "working around it: `make bake` (python scripts/bake_models.py) writes "
            "it to NLTK_DATA "
            f"({os.environ.get('NLTK_DATA') or 'unset — nltk default search path'})."
        ) from exc
    return fn


@lru_cache(maxsize=1)
def concepts(pack: str) -> tuple[ConceptRef, ...]:
    """One concept per frame, its definition the text that gets embedded.

    A frame whose definition is empty after markup stripping is skipped: a
    concept without a definition has no index entry to be retrieved by.
    """
    return tuple(
        ConceptRef(uri=f"fn:{frame.name}", label=frame.name, definition=definition, pack=pack)
        for frame in _framenet().frames()
        if (definition := _clean(frame.definition))
    )


@lru_cache(maxsize=1)
def predicates(pack: str) -> tuple[PredicateRef, ...]:
    """One predicate per core frame element, its domain the frame it belongs to."""
    return tuple(
        PredicateRef(
            id=f"fn:{frame.name}:{name}",
            label=name,
            definition=definition,
            canonical=name.replace("_", " ").lower(),
            domain=f"fn:{frame.name}",
            pack=pack,
        )
        for frame in _framenet().frames()
        for name, element in frame.FE.items()
        if element.coreType == _CORE and (definition := _clean(element.definition))
    )


__all__ = ["concepts", "predicates"]
