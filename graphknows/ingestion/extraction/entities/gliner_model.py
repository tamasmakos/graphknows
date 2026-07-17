"""Adapter boundary for loading the graphgen relation-extraction model."""

from __future__ import annotations

import os
import threading
from typing import Any

# knowledgator/gliner-relex-large-v1.0: joint entity+relation extraction in one
# forward pass (~467M params). Unlike GLiNER2 — which only ever *corroborated* an
# SVO triple under a supplied predicate label — this returns typed head/tail spans
# with the relation and a score, so the extraction itself IS the fact. Relation
# labels are still supplied per chunk (the SVO-mined predicate vocabulary), which
# is the part of the old design that carried its weight.
_RELEX_MODEL = os.environ.get(
    "GRAPHKNOWS_RELEX_MODEL", "knowledgator/gliner-relex-large-v1.0"
)


class _SerialInference:
    """Thread-safety wrapper around the shared extraction model.

    ONE model instance is shared by every concurrently-ingesting conversation
    (they run in one MCP server process and dispatch through asyncio.to_thread),
    and by two independent callers — the extractor and the frame-SRL pass. Its
    ``inference()`` is not reentrant: under 3-way ingest it raised a bare,
    message-less exception that reached the caller as "Error executing tool
    memory_ingest: " with nothing after the colon, losing a document that ingests
    cleanly on its own.

    The lock lives HERE, with the shared object, rather than in a caller: a caller
    -side lock is bypassed by anyone who reaches the model through the
    ``extractor.gliner`` property, and two nested caller locks would deadlock.
    Inference is the ingest bottleneck regardless, so serialising it forfeits
    little — the parallelism that pays is the embedding and graph-write work on
    either side of it.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = threading.Lock()

    def inference(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.inference(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Anything that is not inference() passes through untouched.
        return getattr(self._model, name)


def load_gliner2_model() -> Any:
    """Load the graphgen relation-extraction model behind the adapter boundary."""
    from gliner import GLiNER

    return _SerialInference(GLiNER.from_pretrained(_RELEX_MODEL))
