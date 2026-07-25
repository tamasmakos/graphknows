"""Ontology term-definition embedding index — cosine-match text to classes/properties.

Structurally the same device as ``channels/_frame_index``: embed every term once
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

Absolute cosines run low under the remote embedder (~0.30-0.39), so callers rank
top-k and treat ``min_sim`` as a permissive floor, not a decision threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

from graphknows.ontology.loader import OntologyTerm, load_ontology_terms

log = logging.getLogger(__name__)

# Label injection is gated by top-k ALONE, not by an absolute cosine. Measured
# against bge-small on conversational text, the top-1 class cosine has a median
# of ~0.20 and a floor of ~0.12 — matching a concrete utterance to an abstract
# class definition is inherently a weak signal. That is acceptable here because
# a candidate label costs nothing: the extractor only emits labels it actually
# finds spans for, so an over-broad candidate set loses precision nowhere.
DEFAULT_MIN_SIM: float = 0.0

# Persisted CHUNK->ONTOLOGY_CLASS tags are a different matter: they are stored
# signal that a retrieval channel later trusts, so they take a real floor.
TAG_MIN_SIM: float = 0.20

# Ontology terms offered to the extractor per chunk. GLiNER's practical label
# ceiling is ~24 including the mined + base inventory, so the ontology gets a
# slice of that budget rather than the whole vocabulary. Ontologies smaller than
# this are passed through whole.
DEFAULT_TOP_K: int = 12

_lock = threading.Lock()
_indexes: dict[str, OntologyIndex] = {}


def _cache_dir() -> Path:
    base = os.environ.get("GRAPHKNOWS_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "graphknows"


def _source_digest(terms: list[OntologyTerm]) -> str:
    """Content hash of the loaded terms — changing an ontology invalidates the cache."""
    payload = "\n".join(sorted(f"{t.kind}\t{t.uri}\t{t.label}\t{t.definition}" for t in terms))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(digest: str) -> str:
    """Invalidation key: embed model + dimension + encoding + ontology content.

    The encoding component is not cosmetic. The key previously captured only
    *which* model and *what* content, never *how* the text was encoded — so when
    the declared model suffix started being applied (see
    ``graphknows.storage.embedder._model_suffix``), the key was unchanged and a
    matrix built by the old encoding was silently reused against vectors from the
    new one. Every score was then meaningless while looking perfectly plausible.
    """
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage.embedder import _model_suffix, embed_dim

    s = GraphKnowsSettings()
    model = re.sub(r"[^A-Za-z0-9._-]", "_", s.embed_model or "local")
    dim = s.embed_dimensions or embed_dim()
    suffix = _model_suffix(s.embed_model or "")
    enc = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:8] if suffix else "plain"
    return f"ontology_{model}_{dim}_{enc}_{digest}"


def _embed_norm(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize, so a dot product is a cosine."""
    from graphknows.storage.embedder import embed

    mat = np.asarray(embed(texts), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized: np.ndarray = mat / norms
    return normalized


def _load_from_disk(key: str, uris: list[str]) -> np.ndarray | None:
    """Load a cached matrix, but only if its rows describe exactly ``uris``.

    The sidecar URI list is not optional bookkeeping: row *i* of the matrix is
    meaningful only against the term list it was built from, so validating the
    row count alone will happily pair a cached matrix with a differently-ordered
    term list and return confident matches for the wrong term.
    """
    d = _cache_dir()
    npy, js = d / f"{key}.npy", d / f"{key}.json"
    if not (npy.exists() and js.exists()):
        return None
    try:
        cached_uris = json.loads(js.read_text(encoding="utf-8"))
        if cached_uris != uris:
            log.info("Ontology index cache term order changed; rebuilding %s", key)
            return None
        mat: np.ndarray = np.load(npy).astype(np.float32)
        if mat.shape[0] == len(uris):
            log.info("Loaded ontology index from cache: %s", npy)
            return mat
    except Exception as exc:
        log.warning("Ontology index cache load failed (%s); rebuilding", exc)
    return None


def _save_to_disk(key: str, uris: list[str], mat: np.ndarray) -> None:
    d = _cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        npy, js = d / f"{key}.npy", d / f"{key}.json"
        tmp_npy, tmp_js = d / f"{key}.npy.tmp", d / f"{key}.json.tmp"
        with open(tmp_npy, "wb") as fh:  # file handle: np.save won't append ".npy"
            np.save(fh, mat)
        tmp_js.write_text(json.dumps(uris), encoding="utf-8")
        os.replace(tmp_npy, npy)  # atomic: readers never see a partial matrix
        os.replace(tmp_js, js)
        log.info("Cached ontology index to %s", npy)
    except OSError as exc:
        log.warning("Ontology index cache write failed (%s); in-process only", exc)


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

    def __init__(self, source: str, terms: list[OntologyTerm]) -> None:
        self.source = source
        self.terms = terms
        self.classes = [t for t in terms if t.kind == "class"]
        self.properties = [t for t in terms if t.kind == "property"]
        self.digest = _source_digest(terms)
        self._class_mat = self._matrix(self.classes, "classes")
        self._property_mat = self._matrix(self.properties, "properties")
        # uri -> row within its own kind's matrix.
        self._rows: dict[str, tuple[str, int]] = {
            **{t.uri: ("class", i) for i, t in enumerate(self.classes)},
            **{t.uri: ("property", i) for i, t in enumerate(self.properties)},
        }

    def _matrix(self, terms: list[OntologyTerm], kind: str) -> np.ndarray:
        if not terms:
            return np.zeros((0, 0), dtype=np.float32)
        key = f"{_cache_key(self.digest)}_{kind}"
        uris = [t.uri for t in terms]
        cached = _load_from_disk(key, uris)
        if cached is not None:
            return cached
        log.info("Building ontology index: embedding %d %s from %s", len(terms), kind, self.source)
        mat = _embed_norm([t.index_text for t in terms])
        _save_to_disk(key, uris, mat)
        return mat

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


def load_ontology_index(source: str) -> OntologyIndex | None:
    """Build (or reuse) the index for an ontology file or folder.

    Args:
        source: Path to an ontology file or folder; empty disables ontology
            injection entirely and returns ``None``.

    Returns:
        The index, or ``None`` when ``source`` is empty or yields no terms. A
        bad path is not fatal: ontology injection is an enrichment, so ingestion
        continues without it rather than failing.
    """
    if not source:
        return None
    cached = _indexes.get(source)
    if cached is not None:
        return cached
    with _lock:
        if (cached := _indexes.get(source)) is not None:  # another thread won
            return cached
        try:
            terms = load_ontology_terms(source)
        except (FileNotFoundError, ValueError):
            log.exception("Ontology source unusable; continuing without it")
            return None
        index = OntologyIndex(source, terms)
        _indexes[source] = index
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
