"""Adapter boundary for loading the graphgen relation-extraction model."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from graphknows.settings import get_settings

log = logging.getLogger(__name__)


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


def _resolve_device() -> str:
    """Device for the relex model: CUDA when available, else CPU.

    ``GRAPHKNOWS_RELEX_DEVICE`` overrides (e.g. "cpu" to force CPU on a GPU box).
    The relex model is a ~467M-param torch nn.Module, so a GPU roughly halves
    its per-chunk inference latency — worthwhile since inference is serialised
    and runs once per chunk.
    """
    override = os.environ.get("GRAPHKNOWS_RELEX_DEVICE", "").strip()
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_gliner2_model() -> Any:
    """Load the graphgen relation-extraction model behind the adapter boundary."""
    from gliner import GLiNER

    # knowledgator/gliner-relex-large-v1.0: joint entity+relation extraction in
    # one forward pass (~467M params). Unlike GLiNER2 — which only ever
    # *corroborated* an SVO triple under a supplied predicate label — this returns
    # typed head/tail spans with the relation and a score, so the extraction itself
    # IS the fact. Relation labels are still supplied per chunk (the SVO-mined
    # predicate vocabulary), which is the part of the old design that carried its
    # weight.
    model = GLiNER.from_pretrained(get_settings().relex_model)
    device = _resolve_device()
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as exc:
            # Falling back to CPU beats failing ingest — but silently is how a
            # GPU box ends up running every extraction on CPU for weeks with no
            # symptom other than "ingest feels slow".
            log.warning("relex model could not move to %s, staying on CPU: %s", device, exc)
    return _SerialInference(model)
