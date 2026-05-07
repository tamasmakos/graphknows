"""Shared constants for all Graphknows services.

These values MUST NOT be redefined in individual services.

Example:
    from graphknows.shared.constants import EMBEDDING_MODEL, EntityLabel
"""
from __future__ import annotations

from enum import Enum

EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384


class EntityLabel(str, Enum):
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
