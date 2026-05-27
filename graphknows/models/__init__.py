# DEPRECATED: graphknows.models is stale pipeline-era code.
# New code must use graphknows.memory.models for DTOs.
# Active legacy callers will be retired in Wave 4.
"""Domain models for the graphknows package.

All Pydantic models inherit from :class:`GraphKnowsBaseModel` or
:class:`ImmutableModel` defined in :mod:`graphknows.models.base`.

Example:
    from graphknows.models import Entity, Chunk, Document, BuildJob
"""

from graphknows.models.base import GraphKnowsBaseModel, ImmutableModel
from graphknows.models.constants import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EntityLabel,
)  # ARCH-62-deferred: migrate to graphknows.domain or graphknows.config
from graphknows.models.entity import Chunk, Edge, Entity, Topic
from graphknows.models.pipeline import (
    BuildJob,
    BuildRequest,
    Document,
    SearchQuery,
    SearchResult,
)
from graphknows.models.session import (
    GraphDelta,
    GraphSnapshot,
    ReasoningStep,
    StreamEvent,
)
from graphknows.models.tools import GraphKnowsTool

__all__ = [
    "EMBEDDING_DIM",
    # Constants
    "EMBEDDING_MODEL",
    "BuildJob",
    "BuildRequest",
    "Chunk",
    # Pipeline models
    "Document",
    "Edge",
    # Entity models
    "Entity",
    "EntityLabel",
    "GraphDelta",
    # Base
    "GraphKnowsBaseModel",
    # Tool primitive
    "GraphKnowsTool",
    "GraphSnapshot",
    "ImmutableModel",
    # Session models
    "ReasoningStep",
    "SearchQuery",
    "SearchResult",
    "StreamEvent",
    "Topic",
]
