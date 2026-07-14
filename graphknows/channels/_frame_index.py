"""Frame-definition embedding index — cosine-match an embedding to FrameNet frames.

Embeds every FrameNet frame once as ``"Frame Name. definition"``, L2-normalizes
the definitions into a matrix, and caches that matrix to disk (``.npy`` + names
``.json``) keyed by the embed model + dimension, so the ~1221 definitions are
embedded once and reused across process starts (re-embedding them on every ingest
via the remote embedder would be prohibitive).

``match_frames`` takes a PRE-COMPUTED embedding (a chunk's ingest embedding, or a
query embedding) and returns the top-k frames by cosine. Absolute cosines are low
under the remote embedder (~0.30-0.39), so callers rank top-k rather than gate on
an absolute threshold.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_lock = threading.Lock()
_names: list[str] | None = None
_matrix: np.ndarray | None = None  # (N, dim) float32, L2-normalized rows


def _cache_dir() -> Path:
    base = os.environ.get("GRAPHKNOWS_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "graphknows"


def _cache_key() -> str:
    """Invalidation key: embed model + output dimension (a config change rebuilds)."""
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage.embedder import embed_dim

    s = GraphKnowsSettings()
    model = re.sub(r"[^A-Za-z0-9._-]", "_", s.embed_model or "local")
    dim = s.embed_dimensions or embed_dim()
    return f"frames_{model}_{dim}"


def _build_matrix() -> tuple[list[str], np.ndarray]:
    from graphknows.channels._framenet import _embed_norm, _frame_def, _framenet

    names = [f.name for f in _framenet().frames()]
    texts = [f"{n.replace('_', ' ')}. {_frame_def(n)}" for n in names]
    log.info("Building frame-definition index: embedding %d frames", len(texts))
    mat = _embed_norm(texts)  # (N, dim) float32, already L2-normalized
    return names, mat


def _load_from_disk(key: str) -> tuple[list[str], np.ndarray] | None:
    d = _cache_dir()
    npy, js = d / f"{key}.npy", d / f"{key}.json"
    if not (npy.exists() and js.exists()):
        return None
    try:
        mat = np.load(npy).astype(np.float32)
        names = json.loads(js.read_text(encoding="utf-8"))
        if len(names) == mat.shape[0]:
            log.info("Loaded frame-definition index from cache: %s", npy)
            return names, mat
    except Exception as exc:  # corrupt/partial cache -> rebuild
        log.warning("Frame index cache load failed (%s); rebuilding", exc)
    return None


def _save_to_disk(key: str, names: list[str], mat: np.ndarray) -> None:
    d = _cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        npy, js = d / f"{key}.npy", d / f"{key}.json"
        tmp_npy, tmp_js = d / f"{key}.npy.tmp", d / f"{key}.json.tmp"
        with open(tmp_npy, "wb") as fh:  # file handle: np.save won't append ".npy"
            np.save(fh, mat)
        tmp_js.write_text(json.dumps(names), encoding="utf-8")
        os.replace(tmp_npy, npy)  # atomic: readers never see a partial matrix
        os.replace(tmp_js, js)
        log.info("Cached frame-definition index to %s", npy)
    except OSError as exc:
        log.warning("Frame index cache write failed (%s); in-process only", exc)


def _ensure(force: bool = False) -> tuple[list[str], np.ndarray]:
    global _names, _matrix
    if _matrix is not None and not force:
        return _names, _matrix  # type: ignore[return-value]
    with _lock:
        if _matrix is not None and not force:
            return _names, _matrix  # type: ignore[return-value]
        key = _cache_key()
        loaded = None if force else _load_from_disk(key)
        if loaded is None:
            names, mat = _build_matrix()
            _save_to_disk(key, names, mat)
        else:
            names, mat = loaded
        _names, _matrix = names, mat
        return names, mat


def match_frames(
    embedding: list[float] | np.ndarray, top_k: int = 5, min_sim: float = 0.0
) -> list[tuple[str, float]]:
    """Top-k FrameNet frames for a PRE-COMPUTED embedding, ranked by cosine.

    ``embedding`` need not be normalized. Returns ``[(frame_name, cosine), ...]``
    with at most ``top_k`` entries and cosine >= ``min_sim``.
    """
    v = np.asarray(embedding, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if v.size == 0 or n == 0.0:  # degenerate vector — skip the index build entirely
        return []
    names, mat = _ensure()
    if v.shape[0] != mat.shape[1]:  # embed dims changed under this key -> rebuild
        names, mat = _ensure(force=True)
        if v.shape[0] != mat.shape[1]:
            return []
    sims = mat @ (v / n)
    k = min(top_k, len(names))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(names[i], float(sims[i])) for i in top if sims[i] >= min_sim]


def match_frames_text(
    text: str, top_k: int = 5, min_sim: float = 0.0
) -> list[tuple[str, float]]:
    """Embed ``text`` then match — convenience for the query side."""
    from graphknows.channels._framenet import _embed_norm

    return match_frames(_embed_norm([text])[0], top_k=top_k, min_sim=min_sim)
