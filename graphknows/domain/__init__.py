"""Domain layer: shared concepts, taxonomy, canonical schema, and business rules."""

from graphknows.domain.taxonomy import (
    EntityGroupConfig,
    TaxonomyConfig,
    get_default_taxonomy,
    load_taxonomy,
)
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
    "EntityGroupConfig",
    "TaxonomyConfig",
    "get_default_taxonomy",
    "load_taxonomy",
    "LABEL_CHUNK",
    "LABEL_DIALOG",
    "LABEL_ENTITY",
    "LABEL_FILE",
    "LABEL_SESSION",
    "LABEL_TOPIC",
    "LABEL_TURN",
    "REL_BELONGS_TO",
    "REL_CO_OCCURS",
    "REL_HAS_CHUNK",
    "REL_HAS_DIALOG",
    "REL_HAS_ENTITY",
    "REL_HAS_FILE",
    "REL_HAS_TURN",
    "REL_RELATES_TO",
    "EntityGroupConfig",
    "TaxonomyConfig",
    "get_default_taxonomy",
    "load_taxonomy",
]
