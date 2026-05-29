"""Domain taxonomy configuration — the single source of truth for entity label schemas.

All entity type names, confidence thresholds, classification rules, and relation types
live in a domain YAML file loaded here.  Python code must not contain hardcoded entity
type names; it reads everything from a ``TaxonomyConfig`` instance.

Usage::

    from graphknows.domain.taxonomy import get_default_taxonomy, load_taxonomy

    taxonomy = get_default_taxonomy()          # bundled generic.yaml
    taxonomy = load_taxonomy("/path/my.yaml")  # custom domain
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EntityGroupConfig(BaseModel):
    """Configuration for a single top-level entity group (e.g. PERSON, ORGANIZATION)."""

    # Canonical sub-labels that belong to this group (used by label-space builder)
    children: list[str] = Field(default_factory=list)
    # Text hints used by select_candidate_labels() for GLiNER2 schema selection
    aliases: list[str] = Field(default_factory=list)
    # Free-text description passed to GLiNER2 schema (schema-guided NER)
    description: str = ""
    # Normalised uppercase tokens used by the rule-based classifier
    classification_tokens: list[str] = Field(default_factory=list)
    # Regex patterns (applied to normalised UPPER_SNAKE form) for the classifier
    classification_patterns: list[str] = Field(default_factory=list)
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
    entity_groups: dict[str, EntityGroupConfig] = Field(default_factory=dict)
    # Hard overrides: normalised entity name → canonical label
    overrides: dict[str, str] = Field(default_factory=dict)
    relation_types: list[str] = Field(default_factory=list)
    # Graph hierarchy metadata (informational; consumed by schema.py and E6 segment builder)
    hierarchy: dict[str, Any] = Field(default_factory=dict)

    def get_min_ner_confidence(self, label: str) -> float:
        """Return the minimum NER confidence threshold for *label* (default 0.0)."""
        group = self.entity_groups.get(label)
        return group.min_ner_confidence if group else 0.0

    def entity_label_keys(self) -> list[str]:
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
            "Set GRAPHKNOWS_TAXONOMY_FILE to a valid absolute path, "
            "or leave it unset to use the bundled generic taxonomy."
        )
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return TaxonomyConfig(**data)


def get_default_taxonomy() -> TaxonomyConfig:
    """Return the bundled ``generic`` taxonomy (loaded from the package YAML).

    This is the default taxonomy used when no ``GRAPHKNOWS_TAXONOMY_FILE`` env var is set.
    To select the EU politics taxonomy instead, set the env var to an absolute path to
    ``taxonomy.eu-politics.yaml``.
    """
    env_path = os.environ.get("GRAPHKNOWS_TAXONOMY_FILE")
    if env_path:
        if not Path(env_path).is_absolute():
            raise ValueError("GRAPHKNOWS_TAXONOMY_FILE must be an absolute path to a YAML file")
        yaml_path = Path(env_path)
    else:
        yaml_path = Path(__file__).parent / "taxonomy.generic.yaml"
    return load_taxonomy(str(yaml_path))


# ---------------------------------------------------------------------------
# Bundled taxonomy discovery
# ---------------------------------------------------------------------------

_BUNDLED_DIR = Path(__file__).parent


def list_bundled_taxonomies() -> list[dict[str, str]]:
    """Return metadata for all bundled taxonomy YAML files.

    Returns a list of dicts, each with ``name`` (the filename short-form
    accepted by :func:`resolve_taxonomy_path`) and ``path`` (absolute path).
    """
    return [{"name": p.name, "path": str(p)} for p in sorted(_BUNDLED_DIR.glob("taxonomy.*.yaml"))]


def resolve_taxonomy_path(taxonomy_file: str) -> str:
    """Resolve a taxonomy file argument to an absolute path.

    Accepts either:
    - An absolute or relative path to a ``.yaml`` file (passed through as-is).
    - A short name like ``"taxonomy.eu-politics.yaml"`` — resolved against the
      bundled domain directory so users don't need to know install paths.

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    candidate = Path(taxonomy_file)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
        return str(candidate)
    # Try relative path from cwd first
    if candidate.exists():
        return str(candidate.resolve())
    # Try bundled directory
    bundled = _BUNDLED_DIR / taxonomy_file
    if bundled.exists():
        return str(bundled)
    raise FileNotFoundError(
        f"Taxonomy '{taxonomy_file}' not found as a path or bundled name. "
        f"Bundled options: {[p['name'] for p in list_bundled_taxonomies()]}"
    )
