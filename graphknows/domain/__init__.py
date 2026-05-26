"""Domain layer: shared concepts, taxonomy, canonical schema, and business rules."""

from graphknows.domain.schema import (
    # Node labels
    LABEL_CHUNK,
    LABEL_DIALOG,
    LABEL_ENTITY,
    LABEL_FILE,
    LABEL_SESSION,
    LABEL_TOPIC,
    LABEL_TURN,
    # Relationship types
    REL_BELONGS_TO,
    REL_CO_OCCURS,
    REL_HAS_CHUNK,
    REL_HAS_DIALOG,
    REL_HAS_ENTITY,
    REL_HAS_FILE,
    REL_HAS_TURN,
    REL_RELATES_TO,
)

__all__ = [
    "LABEL_CHUNK",
    "LABEL_DIALOG",
    "LABEL_ENTITY",
    "LABEL_FILE",
    # Node labels
    "LABEL_SESSION",
    "LABEL_TOPIC",
    "LABEL_TURN",
    "REL_BELONGS_TO",
    "REL_CO_OCCURS",
    "REL_HAS_CHUNK",
    "REL_HAS_DIALOG",
    "REL_HAS_ENTITY",
    # Relationship types
    "REL_HAS_FILE",
    "REL_HAS_TURN",
    "REL_RELATES_TO",
]
