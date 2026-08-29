"""Ontology term-definition embedding index — cosine-match text to classes/properties.

Structurally the same device as ``frames/index.py``: embed every term once
as ``"Label. definition"``, L2-normalize into a matrix, cache it to disk so the
definitions are embedded once and reused across process starts. Matching takes a
PRE-COMPUTED embedding (a chunk's ingest embedding, or a query embedding), so
ontology injection costs no extra embedding call on the hot path.

It differs from the frame index in two ways that matter:

- **Not a process singleton.** FrameNet is one global inventory; ontologies vary
  per ingestion and per tenant, so indexes are cached by source-content hash and
  several can be live at once.
- **Two matrices.** Classes steer entity labels, properties steer relation
  predicates, and they must never be mixed into one ranking.

Absolute cosines are NOT calibrated, and the floors below are volume controls,
not decision thresholds. Measured over 20 real LoCoMo turns against the bundled
CCO digest with bge-small, the whole 1437-class space spans 0.168-0.683 (median
0.409, p95 0.493) and a high cosine is not a correctness signal — a thank-you
turn's best class is ``Artifact History`` at 0.545, above the median that a
correctly-matched shopping turn's third-best (``Warehouse``, 0.628) also clears.
Rank top-k, keep the floor above the space's own median, and do not read a high
cosine as a correct term.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from functools import cached_property

import numpy as np

from graphknows.symbolic.index import cache_key, cached_matrix
from graphknows.symbolic.ontology.loader import OntologyTerm, load_ontology

log = logging.getLogger(__name__)

# Fallback floor for direct ``match_*`` callers. The label-hint path passes
# ``settings.ontology_hint_min_sim`` explicitly and that setting carries the
# measurement; the two are pinned equal by
# ``tests/ontology/test_cco_digest.py``.
#
# This used to be 0.0 — "gated by top-k alone" — which was harmless while the
# only bundled ontology had 11 classes, so top-12 WAS the whole ontology. The
# default ontology is now CCO's 1437 classes, where an unfloored top-12 hands
# twelve arbitrary labels to every chunk, including "Thanks, Gina!".
DEFAULT_MIN_SIM: float = 0.46

# Persisted CHUNK->ONTOLOGY_CLASS tags are a different matter: they are stored
# signal that a retrieval channel later trusts, so they take a stricter floor
# than the hints. It must sit above the class space's own p95, or it is not a
# floor at all — at CCO's median (0.409) every chunk is tagged regardless of
# match. 0.52 clears the measured p95 (0.493 over 20 real LoCoMo conv-30 turns
# with bge-small), so a tag has to be an outlier match rather than an average
# one; measured cost is 1/20 chunks losing every tag.
TAG_MIN_SIM: float = 0.52

# Ontology terms offered to the extractor per chunk. GLiNER's practical label
# ceiling is ~24 including the mined + base inventory, so the ontology gets a
# slice of that budget rather than the whole vocabulary. Ontologies smaller than
# this are passed through whole.
DEFAULT_TOP_K: int = 12

_lock = threading.Lock()
# Keyed on (source, on-disk fingerprint) -- see _fingerprint / load_ontology_index.
_indexes: dict[tuple[str, str], OntologyIndex] = {}


def _source_digest(terms: list[OntologyTerm]) -> str:
    """Content hash of the loaded terms — changing an ontology invalidates the cache."""
    payload = "\n".join(
        sorted(
            f"{t.kind}\t{t.uri}\t{t.label}\t{t.definition}\t"
            f"{','.join(t.alt_labels)}\t{','.join(t.examples)}"
            for t in terms
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(digest: str) -> str:
    """Invalidation key: the embedder's identity plus the ontology's content.

    The encoding component (in :func:`~graphknows.symbolic.index.cache_key`) is
    not cosmetic. The key once captured only *which* model and *what* content,
    never *how* the text was encoded — so when the declared model suffix started
    being applied, the key was unchanged and a matrix built by the old encoding
    was silently reused against vectors from the new one. Every score was then
    meaningless while looking perfectly plausible.
    """
    return cache_key("ontology", digest)


def _embed_norm(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize, so a dot product is a cosine."""
    from graphknows.storage.embedder import embed

    mat = np.asarray(embed(texts), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized: np.ndarray = mat / norms
    return normalized


def _top_k(
    mat: np.ndarray,
    terms: list[OntologyTerm],
    embedding: list[float] | np.ndarray,
    top_k: int,
    min_sim: float,
) -> list[tuple[OntologyTerm, float]]:
    """Rank ``terms`` against ``embedding`` by cosine over the normalized ``mat``."""
    if not terms or mat.size == 0:
        return []
    v = np.asarray(embedding, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if v.size == 0 or n == 0.0 or v.shape[0] != mat.shape[1]:
        return []
    sims = mat @ (v / n)
    k = min(top_k, len(terms))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(terms[i], float(sims[i])) for i in idx if sims[i] >= min_sim]


class OntologyIndex:
    """Embedded class/property definitions for one ontology source."""

    def __init__(
        self, source: str, terms: list[OntologyTerm], *, dropped_expressions: int = 0
    ) -> None:
        self.source = source
        self.terms = terms
        self.dropped_expressions = dropped_expressions
        self.classes = [t for t in terms if t.kind == "class"]
        self.properties = [t for t in terms if t.kind == "property"]
        self.digest = _source_digest(terms)
        # uri -> row within its own kind's matrix.
        self._rows: dict[str, tuple[str, int]] = {
            **{t.uri: ("class", i) for i, t in enumerate(self.classes)},
            **{t.uri: ("property", i) for i, t in enumerate(self.properties)},
        }

    # Built on first match, not in __init__. Embedding a vocabulary costs minutes
    # (CCO is 1437 definitions), and most readers of an index never match against
    # it at all — they read labels, domains and ranges. Eager building made every
    # construction pay for a matrix the caller usually never touched, which is
    # what put a 1437-definition embed behind a unit test that injects a stub.
    @cached_property
    def _class_mat(self) -> np.ndarray:
        return self._matrix(self.classes, "classes")

    @cached_property
    def _property_mat(self) -> np.ndarray:
        return self._matrix(self.properties, "properties")

    def _matrix(self, terms: list[OntologyTerm], kind: str) -> np.ndarray:
        if not terms:
            return np.zeros((0, 0), dtype=np.float32)
        key = f"{_cache_key(self.digest)}_{kind}"
        uris = [t.uri for t in terms]

        def _build() -> np.ndarray:
            log.info(
                "Building ontology index: embedding %d %s from %s", len(terms), kind, self.source
            )
            return _embed_norm([t.index_text for t in terms])

        return cached_matrix(key, uris, _build)

    def match_classes(
        self,
        embedding: list[float] | np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        min_sim: float = DEFAULT_MIN_SIM,
    ) -> list[tuple[OntologyTerm, float]]:
        """Top-k ontology classes for a pre-computed embedding, ranked by cosine."""
        return _top_k(self._class_mat, self.classes, embedding, top_k, min_sim)

    def match_properties(
        self,
        embedding: list[float] | np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        min_sim: float = DEFAULT_MIN_SIM,
    ) -> list[tuple[OntologyTerm, float]]:
        """Top-k ontology properties for a pre-computed embedding, ranked by cosine."""
        return _top_k(self._property_mat, self.properties, embedding, top_k, min_sim)

    def term_embedding(self, uri: str) -> list[float]:
        """The term's definition embedding from the built matrix (never a re-embed)."""
        located = self._rows.get(uri)
        if located is None:
            return []
        kind, row = located
        mat = self._class_mat if kind == "class" else self._property_mat
        if row >= mat.shape[0]:
            return []
        return [float(x) for x in mat[row]]

    @property
    def class_labels(self) -> list[str]:
        """Every class label — the vocabulary an entity type can map onto."""
        return [t.label for t in self.classes]

    @property
    def property_labels(self) -> list[str]:
        """Every property label — the vocabulary a relation predicate maps onto."""
        return [t.label for t in self.properties]


def _fingerprint(source: str) -> str:
    """Cheap on-disk identity for ``source``: every file's ``(mtime_ns, size)``.

    ``_indexes`` used to be keyed on the source PATH alone, so an in-place
    ontology upgrade (same path, new content -- exactly how ``rebind_memory``
    is meant to be used) kept resolving to the OLD index for the rest of the
    process: a rebind re-grounded against the ontology it was supposed to
    replace and still reported success (issue #145). Stat-ing every file costs
    microseconds; re-parsing the ontology on every lookup would not.
    """
    from graphknows.symbolic.ontology import rdf_io
    from graphknows.symbolic.ontology.loader import _ontology_files

    try:
        resolved = rdf_io.local_path(source)
        files = _ontology_files(resolved) if resolved.exists() else []
        stat = sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    except (OSError, ValueError):
        return ""
    return hashlib.sha256(repr(stat).encode("utf-8")).hexdigest()[:16]


def load_ontology_index(source: str) -> OntologyIndex | None:
    """Build (or reuse) the index for an ontology file or folder.

    Ontology injection is always on: ``settings.ontology_source`` defaults to
    the bundled schema.org digest, so ``None`` here means either a deliberate
    ``GRAPHKNOWS_ONTOLOGY=""`` or a genuine load failure. Both are logged —
    silently returning ``None`` would make "switched off" and "broken"
    indistinguishable from outside.

    Args:
        source: Path to an ontology file or folder; empty disables ontology
            injection entirely and returns ``None``.

    Returns:
        The index, or ``None`` when ``source`` is empty or yields no terms. A
        bad path is not fatal: ontology injection is an enrichment, so ingestion
        continues without it rather than failing.
    """
    if not source:
        log.warning(
            "Ontology injection is OFF: no source configured (GRAPHKNOWS_ONTOLOGY is empty)"
        )
        return None
    # Cached on (path, on-disk fingerprint): a file replaced in place mints a
    # new key, so the stale entry is simply never looked up again. It is left
    # in ``_indexes`` rather than evicted -- unbounded only in the pathological
    # case of the same process rebinding to many distinct ontology contents,
    # which is not a case this package's callers create.
    key = (source, _fingerprint(source))
    cached = _indexes.get(key)
    if cached is not None:
        return cached
    with _lock:
        if (cached := _indexes.get(key)) is not None:  # another thread won
            return cached
        try:
            loaded = load_ontology(source)
        except (FileNotFoundError, ValueError):
            log.exception("Ontology source unusable; continuing without it")
            return None
        index = OntologyIndex(source, loaded.terms, dropped_expressions=loaded.dropped_expressions)
        _indexes[key] = index
        return index


def reset_ontology_indexes() -> None:
    """Drop the process-wide index cache (tests)."""
    with _lock:
        _indexes.clear()


__all__ = [
    "DEFAULT_MIN_SIM",
    "DEFAULT_TOP_K",
    "TAG_MIN_SIM",
    "OntologyIndex",
    "load_ontology_index",
    "reset_ontology_indexes",
]
