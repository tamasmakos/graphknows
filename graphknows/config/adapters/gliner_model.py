"""Adapter boundary for loading the GLiNER2 graphgen model."""

from __future__ import annotations

from typing import Any


def load_gliner2_model() -> Any:
    """Load the graphgen GLiNER2 model behind the config adapter boundary."""
    from graphknows.extraction.entities.gliner import get_gliner2_model

    return get_gliner2_model()
