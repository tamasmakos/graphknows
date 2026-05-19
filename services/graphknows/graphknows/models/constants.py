"""Shared constants for all graphknows components.

These values must not be redefined in individual modules.

Example:
    from graphknows.models.constants import EMBEDDING_MODEL, EntityLabel
"""

from __future__ import annotations

from enum import StrEnum

EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384


class EntityLabel(StrEnum):
    """LKIF-derived entity labels used across all pipeline stages."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    POLICY = "Policy"
    TREATY = "Treaty"
    LEGISLATION = "Legislation"
    CONCEPT = "Concept"
    EVENT = "Event"
    PLACE = "Place"
    REGULATION = "Regulation"
    TOPIC = "Topic"
