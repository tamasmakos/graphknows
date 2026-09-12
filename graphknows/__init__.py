"""graphknows — agentic memory as an importable library.

Any agent project can integrate graphknows without HTTP, hardcoded endpoints,
or framework-specific glue. :class:`Memory` is the single transport-neutral
entrypoint for all memory operations::

    from graphknows import Memory

    mem = Memory()                                    # builds from env / .env
    await mem.add(messages, user_id="u1", run_id="r7")
    hits = await mem.search("what did she decide?", user_id="u1")
    await mem.flush()                                  # identity + dead-weight report

The same class backs the MCP stdio server, so every transport sees identical
behaviour. For out-of-process use over MCP, see
:mod:`graphknows.integrations.client`.
"""

from __future__ import annotations

import logging

from graphknows._version import __version__
from graphknows.exceptions import (
    ConfigurationError,
    GraphKnowsError,
    MissingExtraError,
    StoreError,
)
from graphknows.memory import Memory
from graphknows.models import (
    MAX_CONTEXT_CHARS,
    Hit,
    IngestResult,
    MemoryScope,
    Message,
    RenderedMemories,
    date_in,
    normalize_messages,
    render_memories,
)
from graphknows.settings import (
    GraphKnowsSettings,
    MemoryMode,
    TopicMode,
)

# A library must not configure logging handlers; attach a NullHandler so records
# are dropped unless the embedding application configures the root logger.
logging.getLogger("graphknows").addHandler(logging.NullHandler())

__all__ = [
    "MAX_CONTEXT_CHARS",
    "ConfigurationError",
    "GraphKnowsError",
    "GraphKnowsSettings",
    "Hit",
    "IngestResult",
    "Memory",
    "MemoryMode",
    "MemoryScope",
    "Message",
    "MissingExtraError",
    "RenderedMemories",
    "StoreError",
    "TopicMode",
    "__version__",
    "date_in",
    "normalize_messages",
    "render_memories",
]
