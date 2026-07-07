"""graphknows infrastructure adapters (port implementations).

The port interfaces themselves live in ``graphknows.ports``. Imports stay lazy
so the baseline GLiNER2 ingest path does not import the DSPy/LiteLLM stack
unless an LLM-assisted relation adapter is explicitly used.
"""

from typing import Any

__all__ = ["DspyRelationExtractorAdapter"]


def __getattr__(name: str) -> Any:
    if name == "DspyRelationExtractorAdapter":
        from graphknows.config.adapters.dspy_relation_extractor import DspyRelationExtractorAdapter

        return DspyRelationExtractorAdapter
    raise AttributeError(name)
