"""Domain taxonomy configuration — the single source of truth for entity label schemas.

All entity type names, confidence thresholds, classification rules, and relation types
live in a domain YAML file loaded here.  Python code must not contain hardcoded entity
type names; it reads everything from a ``TaxonomyConfig`` instance.

Usage::

    from graphknows.graphgen.config.taxonomy import get_default_taxonomy, load_taxonomy

    taxonomy = get_default_taxonomy()          # bundled eu-politics.yaml
    taxonomy = load_taxonomy("/path/my.yaml")  # custom domain
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class EntityGroupConfig(BaseModel):
    """Configuration for a single top-level entity group (e.g. PERSON, ORGANIZATION)."""

    # Canonical sub-labels that belong to this group (used by label-space builder)
    children: List[str] = Field(default_factory=list)
    # Text hints used by select_candidate_labels() for GLiNER2 schema selection
    aliases: List[str] = Field(default_factory=list)
    # Free-text description passed to GLiNER2 schema (schema-guided NER)
    description: str = ""
    # Normalised uppercase tokens used by the rule-based classifier
    classification_tokens: List[str] = Field(default_factory=list)
    # Regex patterns (applied to normalised UPPER_SNAKE form) for the classifier
    classification_patterns: List[str] = Field(default_factory=list)
    # Minimum GLiNER2 confidence required for an entity to be relation-eligible
    min_ner_confidence: float = 0.0

    def matches_name(self, normalised_name: str) -> bool:
        """Return True if *normalised_name* matches any classification rule for this group.

        Args:
            normalised_name: Name already normalised to UPPER_SNAKE form
                (e.g. ``"EUROPEAN_COMMISSION"``).
        """
        if any(token in normalised_name for token in self.classification_tokens):
            return True
        return any(re.search(pat, normalised_name) for pat in self.classification_patterns)


class TaxonomyConfig(BaseModel):
    """Complete domain taxonomy configuration loaded from YAML."""

    name: str
    version: str = "1.0"
    entity_groups: Dict[str, EntityGroupConfig] = Field(default_factory=dict)
    # Hard overrides: normalised entity name → canonical label
    overrides: Dict[str, str] = Field(default_factory=dict)
    relation_types: List[str] = Field(default_factory=list)
    # Graph hierarchy metadata (informational; consumed by schema.py and E6 segment builder)
    hierarchy: Dict[str, Any] = Field(default_factory=dict)

    def get_min_ner_confidence(self, label: str) -> float:
        """Return the minimum NER confidence threshold for *label* (default 0.0)."""
        group = self.entity_groups.get(label)
        return group.min_ner_confidence if group else 0.0

    def entity_label_keys(self) -> List[str]:
        """Return top-level entity group names (e.g. ['PERSON', 'ORGANIZATION', ...])."""
        return list(self.entity_groups.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_taxonomy(path: str) -> TaxonomyConfig:
    """Load and validate a taxonomy from a YAML file at *path*.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {path}. "
            "Set GRAPHGEN_TAXONOMY_FILE to a valid path, "
            "or leave it unset to use the bundled eu-politics taxonomy."
        )
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return TaxonomyConfig(**data)


def get_default_taxonomy() -> TaxonomyConfig:
    """Return the bundled ``eu-politics`` taxonomy (loaded from the package YAML).

    This is the default taxonomy used when no ``GRAPHGEN_TAXONOMY_FILE`` env var is set.
    """
    yaml_path = Path(__file__).parent / "taxonomy.eu-politics.yaml"
    return load_taxonomy(str(yaml_path))
