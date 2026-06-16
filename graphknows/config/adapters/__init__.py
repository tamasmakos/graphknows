"""graphknows adapters: ports and infrastructure adapters.

Imports stay lazy so the baseline GLiNER2 ingest path does not import the
DSPy/LiteLLM stack unless an LLM-assisted relation adapter is explicitly used.
"""

from typing import Any

from graphknows.config.adapters.ports.embedder import IEmbedder
from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor
from graphknows.config.adapters.ports.vector_db import IVectorDb

__all__ = [
    "DspyRelationExtractorAdapter",
    "IEmbedder",
    "IRelationExtractor",
    "IVectorDb",
]


def __getattr__(name: str) -> Any:
    if name == "DspyRelationExtractorAdapter":
        from graphknows.config.adapters.dspy_relation_extractor import DspyRelationExtractorAdapter

        return DspyRelationExtractorAdapter
    raise AttributeError(name)
