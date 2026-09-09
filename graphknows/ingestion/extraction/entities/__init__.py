"""Decoder selection — the one place the memory mode forks.

``build_decoder(settings)`` returns the ``LLMDecoder`` under ``Decoder.llm`` and
the process-wide ``GLiNER2EntityExtractor`` otherwise. Both arms are imported
lazily, so the local stack's models never load in the assisted mode (FR-008).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from graphknows.settings import Decoder

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings

# The extractor owns a ~2.5 GB relation-extraction model. It is expensive to
# build, STATELESS across calls, and its inference() is already serialised
# (gliner_model._SerialInference) — so exactly one belongs per PROCESS.
#
# It used to be constructed per caller, and the callers are per-REQUEST: Memory
# .add_memory() builds a fresh STMService — and therefore a fresh extractor — on
# every ingest call, and Memory itself is cached per NAMESPACE (up to 64). So a
# single-process run rebuilt the whole model once per document, churning 2.5 GB
# each time. The server climbed to ~12.5 GB RSS and died ~28 documents into a
# 10-conversation run, and every document paid ~15s re-loading a model it already
# had. Neither symptom was a leak: it was an object-lifetime mistake — a
# process-wide singleton reached through a per-request object graph.
#
# Cached HERE, at the composition boundary, rather than at the call sites: the
# call sites are request handlers and have no business knowing the resource is
# expensive, and there are several of them (STM service, corpus workflow).
_extractor: Any | None = None
_extractor_lock = threading.Lock()


def build_decoder(settings: GraphKnowsSettings) -> Any:
    """Return the decoder the mode asks for.

    Under ``Decoder.llm`` that is a fresh ``LLMDecoder`` — it is cheap (one
    provider handle, no local model) and carries per-run abstention counters, so
    it is not memoised. Otherwise it is the shared :func:`local_decoder`.
    """
    if settings.decoder is Decoder.llm:
        from graphknows.ingestion.extraction.llm.decoder import LLMDecoder

        return LLMDecoder(settings)
    return local_decoder()


def local_decoder() -> Any:
    """The process-wide GLiNER2 extractor.

    The model is stateless and its inference is lock-serialised, so concurrent
    callers share one instance safely.
    """
    global _extractor
    if _extractor is None:
        with _extractor_lock:
            if _extractor is None:  # double-checked: concurrent first callers
                from graphknows.ingestion.extraction.entities.extractor import (
                    GLiNER2EntityExtractor,
                )

                _extractor = GLiNER2EntityExtractor()
    return _extractor


def reset_entity_extractor() -> None:
    """Drop the cached extractor (tests; releases the model)."""
    global _extractor
    with _extractor_lock:
        _extractor = None


__all__ = ["build_decoder", "local_decoder", "reset_entity_extractor"]
