"""The definition-embedding index every symbolic channel needs.

A channel matches text against its vocabulary by embedding each term's
DEFINITION — never its bare label, which is short, polysemous and matches
badly — into one L2-normalised matrix, caching that matrix to disk, and ranking
by cosine. FrameNet and the ontology each grew their own copy of this: the same
atomic `.npy` + `.json` sidecar layout, the same cache-key construction, the
same cosine top-k.

They differed in one way that mattered, and the stricter behaviour is the one
kept here: the ontology cache validated the sidecar id list **exactly**, while
the frame cache only compared its length. Row *i* is meaningful only against the
id list it was built from, so a length check will happily pair a cached matrix
with a reordered vocabulary and return confident matches for the wrong term.

What is deliberately NOT shared is lifetime. FrameNet's vocabulary is a process
global (one FrameNet per process), the ontology's belongs to a namespace's
loaded scheme. Each keeps its own caching, and borrows only the mechanics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def cache_dir() -> Path:
    """Where built matrices are cached between processes."""
    base = os.environ.get("GRAPHKNOWS_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "graphknows"


def cache_key(prefix: str, *parts: str) -> str:
    """Invalidation key: the embedder's identity, plus whatever *parts* add.

    The encoding component is not redundant with the model name. The key must
    change when the *way* text is encoded changes, not only when the model does
    — otherwise a matrix built under the old encoding is silently reused against
    vectors produced by the new one. See ``storage.embedder.model_suffix``.
    """
    from graphknows.settings import get_settings
    from graphknows.storage.embedder import embed_dim, model_suffix

    s = get_settings()
    model = re.sub(r"[^A-Za-z0-9._-]", "_", s.embed_model or "local")
    dim = s.embed_dimensions or embed_dim()
    suffix = model_suffix(s.embed_model or "")
    enc = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:8] if suffix else "plain"
    return "_".join([prefix, model, str(dim), enc, *parts])


def load_matrix(key: str, ids: list[str]) -> np.ndarray | None:
    """A cached matrix, but only if its rows describe exactly *ids*.

    Returns None — meaning "rebuild" — on a missing, corrupt, or mismatched
    cache. Never raises: a bad cache is a slow start, not a failure.
    """
    npy, js = cache_dir() / f"{key}.npy", cache_dir() / f"{key}.json"
    if not (npy.exists() and js.exists()):
        return None
    try:
        cached_ids = json.loads(js.read_text(encoding="utf-8"))
        if cached_ids != ids:
            log.info("index cache %s describes a different vocabulary; rebuilding", key)
            return None
        mat: np.ndarray = np.load(npy).astype(np.float32)
        if mat.shape[0] == len(ids):
            log.info("loaded index from cache: %s", npy)
            return mat
        log.warning(
            "index cache %s has %d rows for %d ids; rebuilding", key, mat.shape[0], len(ids)
        )
    except Exception as exc:  # corrupt or partial cache -> rebuild
        log.warning("index cache load failed (%s); rebuilding", exc)
    return None


def save_matrix(key: str, ids: list[str], mat: np.ndarray) -> None:
    """Cache *mat* and its id list, atomically.

    Written to temporaries and ``os.replace``d so a reader never sees a partial
    matrix, and never a matrix paired with the wrong sidecar.
    """
    d = cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        npy, js = d / f"{key}.npy", d / f"{key}.json"
        tmp_npy, tmp_js = d / f"{key}.npy.tmp", d / f"{key}.json.tmp"
        with open(tmp_npy, "wb") as fh:  # file handle: np.save won't append ".npy"
            np.save(fh, mat)
        tmp_js.write_text(json.dumps(ids), encoding="utf-8")
        os.replace(tmp_npy, npy)
        os.replace(tmp_js, js)
        log.info("cached index to %s", npy)
    except OSError as exc:
        log.warning("index cache write failed (%s); in-process only", exc)


def cached_matrix(key: str, ids: list[str], build: Callable[[], np.ndarray]) -> np.ndarray:
    """Load the matrix for *ids*, or build it with *build* and cache the result."""
    cached = load_matrix(key, ids)
    if cached is not None:
        return cached
    mat = build()
    save_matrix(key, ids, mat)
    return mat


def match(
    embedding: list[float] | np.ndarray,
    ids: list[str],
    mat: np.ndarray,
    top_k: int,
    min_sim: float,
) -> list[tuple[str, float]]:
    """Top-*k* ids for a pre-computed embedding, ranked by cosine, above *min_sim*.

    *embedding* need not be normalised; *mat* must have L2-normalised rows, so
    the dot product IS the cosine.

    A degenerate (empty or zero-norm) vector returns nothing rather than ranking
    the whole vocabulary by numerical noise.
    """
    v = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if v.size == 0 or norm == 0.0 or mat.size == 0:
        return []
    if v.shape[0] != mat.shape[1]:
        log.warning(
            "query dim %d does not match index dim %d; no match is possible",
            v.shape[0],
            mat.shape[1],
        )
        return []
    sims = mat @ (v / norm)
    order = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in order if float(sims[i]) >= min_sim]
