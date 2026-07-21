"""graphknows — agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue. :class:`Memory` is the single transport-neutral
entrypoint for all memory operations::

    from graphknows import Memory

    mem = Memory()                                    # builds from env / .env
    await mem.add(messages, user_id="u1", run_id="r7")
    hits = await mem.search("what did she decide?", user_id="u1")
    await mem.flush(user_id="u1")                      # STM -> LTM promotion

The same class backs the MCP stdio server and the web Memory Mirror, so every
transport sees identical behaviour. For out-of-process use over MCP, see
:mod:`graphknows.integrations.client`; to implement a custom storage backend,
see :mod:`graphknows.storage.base`.
"""

from __future__ import annotations

import logging

from graphknows._version import __version__
from graphknows.exceptions import (
    ConfigurationError,
    ExtractionError,
    GraphKnowsError,
    IngestError,
    MissingExtraError,
    RetrievalError,
    StoreError,
)
from graphknows.memory import Memory
from graphknows.models import (
    IngestResult,
    MemoryScope,
    Message,
    normalize_messages,
)
from graphknows.settings import (
    GraphKnowsSettings,
    MemoryMode,
    RetrievalStrategy,
    TopicMode,
)

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
    "MemoryScope",
    "Message",
    "MissingExtraError",
    "RetrievalError",
    "RetrievalStrategy",
    "StoreError",
    "TopicMode",
    "__version__",
    "normalize_messages",
]
