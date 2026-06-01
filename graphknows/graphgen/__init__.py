"""graphknows.graphgen — graph-generation primitives.

This package provides the extraction and chunking primitives shared by both
the STM-ingest path and the LTM-flush promotion path.  No orchestration or
legacy batch-pipeline code lives here.

Public API::

    from graphknows.graphgen import (
        chunk_text,
        get_gliner2_model,
        DSPyExtractor,
        BaseExtractor,
        build_top_level_label_space,
    )
"""

from __future__ import annotations

from graphknows.graphgen.chunking import process_document_splitting as chunk_text
from graphknows.graphgen.extraction.gliner import get_gliner2_model
from graphknows.graphgen.extraction.ontology import build_top_level_label_space
from graphknows.graphgen.extraction.relations import BaseExtractor, DSPyExtractor

__all__ = [
    "BaseExtractor",
    "DSPyExtractor",
    "build_top_level_label_space",
    "chunk_text",
    "get_gliner2_model",
]
