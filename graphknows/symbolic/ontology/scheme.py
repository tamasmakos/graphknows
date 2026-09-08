"""The process-wide lexicalized SKOS scheme, and the corpus statistics it needs.

Building the scheme parses a 600 KB digest and lemmatises every authored
altLabel, so it is built once and cached — the same reasoning as
``catalog.load_ontology_index``, which caches the embedding matrices.

The corpus statistics are the part that cannot be cached globally. Label
selection drops altLabels that are ubiquitous IN THIS CORPUS (see
``lexical.usable_index``), because a word appearing in most chunks discriminates
between none of them. That is a per-namespace fact, so it is computed from the
turns a session actually ingests and held per namespace.
"""

from __future__ import annotations

import logging
import threading
from functools import cache
from pathlib import Path

from graphknows.symbolic.ontology.lexical import document_frequency, usable_index
from graphknows.symbolic.ontology.skos import Concept, build_scheme, coarse_aliases, extend_scheme

log = logging.getLogger(__name__)

_lock = threading.Lock()
_schemes: dict[tuple[str, str], dict[str, Concept]] = {}
_indexes: dict[str, dict[str, set[str]]] = {}


def load_scheme(source: str, overlay: str = "") -> dict[str, Concept]:
    """The lexicalized scheme for ``source``, plus an optional input overlay.

    ``overlay`` is the seam that makes this ontology-DRIVEN rather than
    hardcoded: a SKOS file supplying the everyday concepts the base ontology
    does not carry, with authored altLabels and ``broader`` links into it.
    """
    key = (source, overlay)
    if (cached := _schemes.get(key)) is not None:
        return cached
    with _lock:
        if (cached := _schemes.get(key)) is not None:  # another thread won
            return cached
        scheme = build_scheme(Path(source))
        if overlay:
            path = Path(overlay)
            if path.exists():
                scheme = extend_scheme(scheme, path)
            else:
                # An overlay that is configured but missing is a broken install,
                # not an absent feature: without it the conversational relations
                # disappear and the graph silently thins out.
                log.error("Ontology overlay %r not found; conversational relations absent", overlay)
        log.info(
            "SKOS scheme: %d concepts from %s%s",
            len(scheme),
            source,
            f" + {overlay}" * bool(overlay),
        )
        _schemes[key] = scheme
        return scheme


@cache
def load_coarse_aliases(overlay: str) -> dict[str, str]:
    """The overlay's curated coarse-type -> CCO-class mapping, cached per path.

    Same caching intent as :func:`load_scheme`: an empty ``overlay`` means no
    curated aliases at all, and a configured-but-missing path is a broken
    install rather than an absent feature. ``lru_cache`` gives thread-safe
    memoisation for a dict this small in one decorator.
    """
    if not overlay:
        return {}
    path = Path(overlay)
    if path.exists():
        return coarse_aliases(path)
    log.error("Ontology overlay %r not found; curated coarse-type aliases absent", overlay)
    return {}


def build_index(
    scheme: dict[str, Concept], corpus: list[str], nlp: object, namespace: str = ""
) -> dict[str, set[str]]:
    """AltLabel -> concepts, with this corpus's ubiquitous lemmas removed.

    ``corpus`` should be one document per TURN, not per window: overlapping
    windows share most of their text, so document frequency computed over them
    reports every content word as ubiquitous and empties the vocabulary.
    """
    if namespace and (cached := _indexes.get(namespace)) is not None:
        return cached
    docs = [nlp(text) for text in corpus if text.strip()]  # type: ignore[operator]
    index = usable_index(scheme, document_frequency(docs), doc_count=len(docs))
    log.info(
        "Lexical index for %s: %d usable altLabels from %d documents",
        namespace or "(unnamed)",
        len(index),
        len(docs),
    )
    if namespace:
        _indexes[namespace] = index
    return index
