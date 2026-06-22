"""graphknows.extraction — the knowledge-extraction engine.

Consolidates entity extraction (GLiNER), relation extraction (spaCy / DSPy
strategies), ontology metadata (RDF loading, routing, filtering), and chunking
behind one package. The STM-ingest and LTM-flush paths both build on these
primitives.

Public API::

    from graphknows.extraction import (
        chunk_text,
        get_gliner2_model,
        DSPyExtractor,
        BaseExtractor,
        build_top_level_label_space,
    )
"""

from __future__ import annotations

from graphknows.extraction.chunking import process_document_splitting as chunk_text
from graphknows.extraction.entities.gliner import get_gliner2_model
from graphknows.extraction.entities.label_space import build_top_level_label_space
from graphknows.extraction.relations.dspy import BaseExtractor, DSPyExtractor

__all__ = [
    "BaseExtractor",
    "DSPyExtractor",
    "build_top_level_label_space",
    "chunk_text",
    "get_gliner2_model",
]
