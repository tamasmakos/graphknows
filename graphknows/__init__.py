"""graphknows — agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue. :class:`Memory` is the single transport-neutral
entrypoint for all memory operations::

    from graphknows import Memory

    mem = Memory()                                 # builds from env / .env
    await mem.ingest_memory("some text", session_id="s1")
    hits = await mem.recall_memory("my query", session_id="s1")

The same class backs the MCP stdio server and the web Memory Mirror, so every
transport sees identical behaviour. For out-of-process use over MCP, see
:mod:`graphknows.client`; to implement a custom storage backend, see
:mod:`graphknows.ports`.
"""

from __future__ import annotations

import logging
from typing import Any

__version__ = "1.0.0"

from graphknows.config import GraphKnowsSettings, MemoryMode
from graphknows.exceptions import (
    ConfigurationError,
    ExtractionError,
    GraphKnowsError,
    IngestError,
    MissingExtraError,
    RetrievalError,
    StoreError,
)
from graphknows.memory.models.ingest import IngestResult
from graphknows.runtime import Memory

# A library must not configure logging handlers; attach a NullHandler so records
# are dropped unless the embedding application configures the root logger.
logging.getLogger("graphknows").addHandler(logging.NullHandler())

__all__ = [
    "ConfigurationError",
    "ExtractionError",
    "GraphKnowsError",
    "GraphKnowsSettings",
    "IngestError",
    "IngestResult",
    "Memory",
    "MemoryMode",
    "MissingExtraError",
    "RetrievalError",
    "StoreError",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Provide the deprecated ``GraphKnowsRuntime`` alias with a warning."""
    if name == "GraphKnowsRuntime":
        import warnings

        warnings.warn(
            "graphknows.GraphKnowsRuntime is deprecated; use graphknows.Memory instead. "
            "The alias will be removed in graphknows 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Memory
    raise AttributeError(f"module 'graphknows' has no attribute {name!r}")
