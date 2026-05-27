"""Short-term memory recall — compatibility shim.

DEPRECATED: Use graphknows.memory.stm.recall directly.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.memory.models.recall import RecallResult
from graphknows.memory.stm._async_bridge import _run as _run
from graphknows.memory.stm.recall import (
    RRF_K,
    _entity_boost,
    _shape_score,
    recall_short_term,
)

log = logging.getLogger("graphknows.retrieval.recall_stm")


def _run_stm_ann(
    pg_store: Any,
    session_id: str,
    embedding: list[float],
    top_k: int,
    results: dict[str, tuple[float, RecallResult]],
) -> None:
    """ANN search stm_chunks + stm_turns — shim that uses local _run so tests can mock it."""
    try:
        chunks = _run(pg_store.search_stm_chunks(session_id, embedding, top_k=top_k * 2))
        for rank, chunk in enumerate(chunks):
            rrf = 1.0 / (RRF_K + rank + 1)
            cid = chunk["id"]
            if cid not in results or results[cid][0] < rrf:
                results[cid] = (
                    rrf,
                    RecallResult(
                        text=chunk["text"],
                        source_type="chunk",
                        scope="short_term",
                        score=rrf,
                        cosine_score=chunk.get("cosine_score", 0.0),
                        entity_names=chunk.get("metadata", {}).get("entity_names", []),
                        heading_path=chunk.get("heading_path", ""),
                        provenance={
                            "session_id": session_id,
                            "file_id": chunk.get("file_id", ""),
                            "memory_class": chunk.get("metadata", {}).get(
                                "memory_class", "raw_evidence"
                            ),
                        },
                        metadata=chunk.get("metadata", {}),
                    ),
                )
    except Exception as exc:
        log.warning("stm_chunks ANN failed: %s", exc)

    try:
        turns = _run(pg_store.search_stm_turns(session_id, embedding, top_k=top_k * 2))
        for rank, turn in enumerate(turns):
            rrf = 1.0 / (RRF_K + rank + 1)
            tid = turn["id"]
            if tid not in results or results[tid][0] < rrf:
                results[tid] = (
                    rrf,
                    RecallResult(
                        text=turn["text"],
                        source_type="turn",
                        scope="short_term",
                        score=rrf,
                        provenance={
                            "session_id": session_id,
                            "role": turn.get("role", ""),
                            "memory_class": "raw_evidence",
                        },
                        metadata=turn.get("metadata", {}),
                    ),
                )
    except Exception as exc:
        log.warning("stm_turns ANN failed: %s", exc)


__all__ = [
    "RRF_K",
    "RecallResult",
    "_entity_boost",
    "_run_stm_ann",
    "_shape_score",
    "recall_short_term",
]
