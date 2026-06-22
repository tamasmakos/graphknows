"""Ontology-driven relation filter — post-extraction curation.

Drops extracted relations whose label is not part of the configured ontology's
property labels. Matching is case- and punctuation-insensitive so that surface
forms like ``member_of``, ``MEMBEROF`` and ``member of`` all map to the ontology
property ``memberOf``.

An empty allowlist disables the filter (pass-through), so extraction modes that
do not constrain relations behave as before.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("graphknows.extraction.ontology.relation_filter")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _normalize(label: str) -> str:
    """Lower-case and strip all non-alphanumeric characters for comparison."""
    return _NON_ALNUM.sub("", str(label).lower())


class RelationOntologyFilter:
    """Keep only relations whose label matches an allowed ontology property.

    Args:
        allowed_labels: Ontology property labels. An empty list yields an
            inactive (pass-through) filter.
    """

    def __init__(self, allowed_labels: list[str]) -> None:
        self._allowed = {_normalize(label) for label in allowed_labels if _normalize(label)}

    @property
    def active(self) -> bool:
        """Whether the filter constrains anything (False = pass-through)."""
        return bool(self._allowed)

    def is_allowed(self, label: str) -> bool:
        """Return True if *label* maps to an allowed property (or filter inactive)."""
        if not self.active:
            return True
        return _normalize(label) in self._allowed

    def filter(self, relations: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
        """Return only relations whose ``label_key`` value is allowed."""
        if not self.active:
            return relations
        return [r for r in relations if self.is_allowed(r.get(label_key, ""))]


def build_relation_filter(ontology_id: str = "schema-org") -> RelationOntologyFilter:
    """Build a relation filter from a bundled ontology's property labels.

    Falls back to an inactive (pass-through) filter when the ontology cannot be
    loaded, so ingestion never fails because of a missing ontology asset.
    """
    from graphknows.extraction.ontology.loader import OntologyLoader

    try:
        manifest = OntologyLoader().load(ontology_id)
    except (KeyError, FileNotFoundError) as exc:
        log.warning("Relation filter inactive — ontology '%s' unavailable: %s", ontology_id, exc)
        return RelationOntologyFilter([])
    return RelationOntologyFilter(manifest.property_labels)
