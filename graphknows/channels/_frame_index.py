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
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Similarity floor for a chunk/query to "evoke" a frame. Absolute cosines run low
# under the remote embedder (~0.30-0.39), so this is a permissive gate applied on
# TOP of top-k: a frame matches when it is both top-k and above this floor.
DEFAULT_MIN_SIM: float = 0.20

# Shared, namespace-independent ArcadeDB catalog of frame definitions.
_CATALOG_DB = "frames"

_lock = threading.Lock()
_names: list[str] | None = None
_matrix: np.ndarray | None = None  # (N, dim) float32, L2-normalized rows


def _cache_dir() -> Path:
    base = os.environ.get("GRAPHKNOWS_CACHE_DIR")
    return Path(base) if base else Path.home() / ".cache" / "graphknows"


def _cache_key() -> str:
    """Invalidation key: embed model + dimension + encoding (a change rebuilds).

    The encoding component mirrors the ontology catalog's: the key must change
    when the *way* text is encoded changes, not only when the model does, or a
    matrix built under the old encoding is reused against new vectors. See
    ``graphknows.storage.embedder._model_suffix``.
    """
    import hashlib

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage.embedder import _model_suffix, embed_dim

    s = GraphKnowsSettings()
    model = re.sub(r"[^A-Za-z0-9._-]", "_", s.embed_model or "local")
    dim = s.embed_dimensions or embed_dim()
    suffix = _model_suffix(s.embed_model or "")
    enc = hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:8] if suffix else "plain"
    return f"frames_{model}_{dim}_{enc}"


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
    embedding: list[float] | np.ndarray, top_k: int = 5, min_sim: float = DEFAULT_MIN_SIM
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
    text: str, top_k: int = 5, min_sim: float = DEFAULT_MIN_SIM
) -> list[tuple[str, float]]:
    """Embed ``text`` then match — convenience for the query side."""
    from graphknows.channels._framenet import _embed_norm

    return match_frames(_embed_norm([text])[0], top_k=top_k, min_sim=min_sim)


def frame_summary(name: str) -> str:
    """The frame's definition text — the FRAME node's ``summary``."""
    from graphknows.channels._framenet import _frame_def

    return _frame_def(name)


def frame_embedding(name: str) -> list[float]:
    """The frame DEFINITION embedding from the runtime index (L2-normalized).

    Reuses the already-built matrix, so populating a FRAME node's embedding at
    ingest is a copy — never a per-chunk re-embed.
    """
    names, mat = _ensure()
    try:
        i = names.index(name)
    except ValueError:
        return []
    return mat[i].tolist()


async def ensure_frame_catalog(client: Any, db: str = _CATALOG_DB, dims: int | None = None) -> int:
    """Build the shared, namespace-independent ``frames`` ArcadeDB catalog (idempotent).

    Vertex ``FRAME_DEF(name UNIQUE, summary, embedding + LSM_VECTOR COSINE)`` populated
    with all FrameNet frames from the in-process matrix (the same source as the disk
    cache), so the ~1221 definitions are embedded once. This is persistent, shareable
    reference data — it is NOT per-namespace and NOT on the ingest/query hot path
    (ingest copies embeddings from the matrix; query vector-searches namespace-local
    FRAME nodes). Returns the FRAME_DEF count.
    """
    from graphknows.storage.arcadedb.graph_store import _emb_sql

    names, mat = _ensure()
    dims = dims or int(mat.shape[1])
    await client.create_database(db)  # idempotent

    # Idempotent guard: skip the (one-time) 1221-row populate if already built.
    try:
        rows = await client.query(db, "SELECT count(*) AS c FROM FRAME_DEF", language="sql")
        if rows and int(rows[0].get("c") or 0) >= len(names):
            return int(rows[0]["c"])
    except Exception as exc:
        # Expected on a first run (FRAME_DEF does not exist yet) — but log it,
        # because the same swallow hides a broken connection, and a "populate"
        # that silently re-runs every call is indistinguishable from one that
        # never worked.
        log.debug("FRAME_DEF idempotency probe failed, populating: %s", exc)

    ddl = [
        "CREATE VERTEX TYPE FRAME_DEF IF NOT EXISTS",
        "CREATE PROPERTY FRAME_DEF.name IF NOT EXISTS STRING",
        "CREATE PROPERTY FRAME_DEF.summary IF NOT EXISTS STRING",
        "CREATE PROPERTY FRAME_DEF.embedding IF NOT EXISTS LIST",
        "CREATE INDEX IF NOT EXISTS ON FRAME_DEF(name) UNIQUE",
        f"CREATE INDEX IF NOT EXISTS ON FRAME_DEF(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{dims},distanceFunction:'COSINE'}}",
    ]
    for stmt in ddl:
        try:
            await client.command(db, stmt, language="sql")
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    # Per-row upsert: the embedding must be an inline literal (LIST params conflict
    # with the LSM_VECTOR index), so it cannot ride a batched UNWIND param — same
    # constraint the golden store honours for CHUNK/ENTITY/TOPIC embeddings.
    for i, name in enumerate(names):
        await client.command(
            db,
            f"MERGE (n:FRAME_DEF {{name: $name}}) "
            f"SET n.summary = $summary, n.embedding = {_emb_sql(mat[i].tolist())}",
            params={"name": name, "summary": frame_summary(name)},
        )
    log.info("Frame catalog built: %d FRAME_DEF in db=%s", len(names), db)
    return len(names)
