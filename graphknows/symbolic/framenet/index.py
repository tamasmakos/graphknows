"""Frame-definition embedding index — cosine-match an embedding to FrameNet frames.

Embeds every FrameNet frame once as its name, definition, frame-element names
and definitions, and lexical-unit names and definitions (see
:func:`_embed_frames`), L2-normalizes the result into a matrix, and caches that
matrix to disk (``.npy`` + names ``.json``) keyed by the embed model + dimension
+ text recipe, so the ~1221 frames are embedded once and reused across process
starts (re-embedding them on every ingest via the remote embedder would be
prohibitive).

``match_frames`` takes a PRE-COMPUTED embedding (a chunk's ingest embedding, or a
query embedding) and returns the top-k frames by cosine. Absolute cosines are low
under the remote embedder (~0.30-0.39), so callers rank top-k rather than gate on
an absolute threshold.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from graphknows.symbolic.index import cache_key, load_matrix, match, save_matrix

log = logging.getLogger(__name__)

# Similarity floor for a chunk/query to "evoke" a frame. Absolute cosines run low
# under the remote embedder (~0.30-0.39), so this is a permissive gate applied on
# TOP of top-k: a frame matches when it is both top-k and above this floor.
DEFAULT_MIN_SIM: float = 0.20

_lock = threading.Lock()
_names: list[str] | None = None
_matrix: np.ndarray | None = None  # (N, dim) float32, L2-normalized rows


def _cache_key() -> str:
    # "fe_lu2" names the text recipe: frame name + definition + LU WORDS + FE
    # text (see _embed_frames). Bumped from "fe_lu" when the LU words moved
    # ahead of the FE definitions in the assembled text (they used to be last
    # and were silently cut for most frames) -- a stale disk cache built under
    # either older recipe must not be reused against this one.
    return cache_key("frames", "fe_lu2")


# Ceiling on the assembled text so the tail (later FEs/LUs) is not silently
# dropped by the encoder's own token limit before it ever sees it.
_MAX_EMBED_CHARS = 1200


# Query widths a forced rebuild already failed to reconcile; see match_frames.
_unreconciled_dims: set[int] = set()


def _frame_names() -> list[str]:
    """The vocabulary, without embedding anything.

    Split from the build deliberately: the cache is keyed on this list, so it
    has to be obtainable BEFORE deciding whether to embed. Fetching it after
    would mean embedding 1200-odd definitions to discover the cache was warm.
    """
    from graphknows.symbolic.framenet.framenet import _framenet

    return [f.name for f in _framenet().frames()]


def _frame_embed_text(name: str) -> str:
    """Name + definition + this frame's LU words + its FE names/definitions.

    LU words come right after the definition, not last: they are a single
    compact line (bare words, not each LU's own definition), so they survive
    the char cap below for virtually every frame, whereas the FE definitions
    that follow are the part actually meant to be sacrificed when a frame has
    too many of them. Putting the verbose part last is what used to drop LU
    signal entirely for 933 of 1221 frames (issue #145) -- the fix is ordering
    the cheap, high-signal part first, not raising the cap.

    Deliberately NOT ``fn.exemplars()`` — a multi-second full-corpus load — and
    the frame/FE/LU definitions already carry the ``<ex>`` example text FrameNet
    embeds inline, so this loses little of what exemplars would add.
    """
    from graphknows.symbolic.framenet.fe import frame_elements
    from graphknows.symbolic.framenet.framenet import _frame_def, _framenet

    parts = [f"{name.replace('_', ' ')}. {_frame_def(name)}"]
    fn = _framenet()
    try:
        lex_units = fn.frame(name).lexUnit
    except Exception:
        lex_units = {}
    words = sorted({lu_name.rpartition(".")[0] or lu_name for lu_name in lex_units})
    if words:
        parts.append("Lexical units: " + ", ".join(words) + ".")
    for fe in frame_elements(name).values():
        parts.append(f"{fe.name}: {fe.definition}")
    # ponytail: hard char cap instead of pulling real exemplar sentences via
    # fn.exemplars() (a multi-second full-corpus load); upgrade to that if
    # frame reranking quality demands richer examples than defs already carry.
    return " ".join(p.strip() for p in parts if p.strip())[:_MAX_EMBED_CHARS]


def _embed_frames(names: list[str]) -> np.ndarray:
    from graphknows.symbolic.framenet.framenet import _embed_norm

    texts = [_frame_embed_text(n) for n in names]
    log.info("Building frame-definition index: embedding %d frames", len(texts))
    return _embed_norm(texts)  # (N, dim) float32, already L2-normalized


def _ensure(force: bool = False) -> tuple[list[str], np.ndarray, bool]:
    """The frame names, their definition matrix, and whether it came from disk.

    Process-global: there is one FrameNet per process, so the matrix outlives
    any single namespace. ``force`` skips the disk cache — used when the
    embedder's dimension no longer matches what the cache holds.

    The third element is what makes the rebuild in ``match_frames`` decidable:
    only a matrix loaded from a STALE CACHE can be reconciled by re-embedding.
    One that was just built came from this process's own embedder, so building
    it a second time cannot produce a different width.
    """
    global _names, _matrix
    if _matrix is not None and not force:
        return _names, _matrix, True  # type: ignore[return-value]
    with _lock:
        if _matrix is not None and not force:
            return _names, _matrix, True  # type: ignore[return-value]
        names = _frame_names()
        key = _cache_key()
        mat = None if force else load_matrix(key, names)
        from_cache = mat is not None
        if mat is None:
            mat = _embed_frames(names)
            save_matrix(key, names, mat)
        _names, _matrix = names, mat
    return names, mat, from_cache


def match_frames(
    embedding: list[float] | np.ndarray, top_k: int = 5, min_sim: float = DEFAULT_MIN_SIM
) -> list[tuple[str, float]]:
    """Top-k FrameNet frames for a PRE-COMPUTED embedding, ranked by cosine.

    ``embedding`` need not be normalized. Returns ``[(frame_name, cosine), ...]``
    with at most ``top_k`` entries and cosine >= ``min_sim``.
    """
    v = np.asarray(embedding, dtype=np.float32).ravel()
    if v.size == 0 or float(np.linalg.norm(v)) == 0.0:
        # Degenerate vector — answer before touching the index, so a bad query
        # never pays for embedding the whole frame vocabulary.
        return []
    if v.shape[0] in _unreconciled_dims:
        return []
    # The embedder's own width, known without embedding anything. A caller whose
    # vectors are not this embedder's can be answered here, before the index is
    # touched at all — otherwise the FIRST such call embeds all ~1221 definitions
    # only to discard them. That cost a fully-mocked unit test 5 minutes of real
    # model work (and a huggingface.co round trip) for a matrix it could never use.
    from graphknows.storage.embedder import EMBED_DIM

    if _matrix is None and v.shape[0] != EMBED_DIM:
        _unreconciled_dims.add(v.shape[0])
        log.warning(
            "query embedding is %d-dimensional but the embedder produces %d; "
            "frame reranking is off for this process",
            v.shape[0],
            EMBED_DIM,
        )
        return []
    names, mat, from_cache = _ensure()
    if mat.size and v.shape[0] != mat.shape[1]:
        # Dims disagree under this cache key. Rebuilding can only help when the
        # matrix came off DISK — a stale cache written by a different embedder.
        # One this process just built came from the same embedder that will build
        # the replacement, so a rebuild re-embeds 1221 definitions to land on the
        # identical width. Retrying per call was worse still: the retrieval suite
        # spent 19 of its 22 minutes in exactly that loop.
        _unreconciled_dims.add(v.shape[0])
        if not from_cache:
            return []
        names, mat, _ = _ensure(force=True)
        if mat.size and v.shape[0] != mat.shape[1]:
            log.warning(
                "frame index is %d-dimensional but the query embedding is %d; "
                "frame reranking is off for this process",
                mat.shape[1],
                v.shape[0],
            )
            return []
    return match(v, names, mat, top_k, min_sim)


def match_frames_text(
    text: str, top_k: int = 5, min_sim: float = DEFAULT_MIN_SIM
) -> list[tuple[str, float]]:
    """Embed ``text`` then match — convenience for the query side."""
    from graphknows.symbolic.framenet.framenet import _embed_norm

    return match_frames(_embed_norm([text])[0], top_k=top_k, min_sim=min_sim)


def frame_summary(name: str) -> str:
    """The frame's definition text — the FRAME node's ``summary``."""
    from graphknows.symbolic.framenet.framenet import _frame_def

    return _frame_def(name)


def frame_embedding(name: str) -> list[float]:
    """The frame DEFINITION embedding from the runtime index (L2-normalized).

    Reuses the already-built matrix, so populating a FRAME node's embedding at
    ingest is a copy — never a per-chunk re-embed.
    """
    names, mat, _ = _ensure()
    try:
        i = names.index(name)
    except ValueError:
        return []
    return mat[i].tolist()
