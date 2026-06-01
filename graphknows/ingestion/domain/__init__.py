"""Domain layer: shared concepts, taxonomy, canonical schema, and business rules."""

from graphknows.ingestion.domain.normalization import standardize_label
from graphknows.ingestion.domain.schema import (
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
from graphknows.ingestion.domain.taxonomy import (
    EntityGroupConfig,
    TaxonomyConfig,
    get_default_taxonomy,
    load_taxonomy,
)

__all__ = [
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
    "EntityGroupConfig",
    "TaxonomyConfig",
    "TaxonomyConfig",
    "get_default_taxonomy",
    "get_default_taxonomy",
    "load_taxonomy",
    "load_taxonomy",
    "standardize_label",
]
