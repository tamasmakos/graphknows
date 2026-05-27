"""Compatibility shim — taxonomy now lives in graphknows.domain.taxonomy."""

# DEPRECATED: import from graphknows.domain.taxonomy directly.
from graphknows.domain.taxonomy import (  # noqa: F401
    EntityGroupConfig,
    TaxonomyConfig,
    load_taxonomy,
    get_default_taxonomy,
)

__all__ = ["EntityGroupConfig", "TaxonomyConfig", "load_taxonomy", "get_default_taxonomy"]
