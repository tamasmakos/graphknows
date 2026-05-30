"""Cypher identifier and property helpers shared by the Neo4j store."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _escape_cypher_identifier(identifier: str) -> str:
    """Escape a Cypher identifier (label or relationship type) with backticks if needed."""
    if not identifier:
        return identifier

    needs_escaping = (
        " " in identifier
        or "-" in identifier
        or not identifier.replace("_", "").isalnum()
        or identifier[0].isdigit()
    )

    if needs_escaping:
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"
    return identifier


def _clean_props_for_neo4j(props: dict[str, Any]) -> dict[str, Any]:
    """Clean a property dict so every value is Neo4j-compatible.

    Neo4j supports primitives (bool, int, float, str) and homogeneous lists of
    primitives. Everything else is JSON-encoded to a string.
    """
    cleaned: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue

        if isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            first = value[0]
            if all(isinstance(x, type(first)) for x in value):
                if isinstance(first, (int, float, str)):
                    cleaned[key] = list(value)
                else:
                    try:
                        cleaned[key] = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError) as exc:
                        logger.debug("Failed to JSON-encode %s: %s", key, exc)
                        cleaned[key] = str(value)
            else:
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError) as exc:
                    logger.debug("Failed to JSON-encode %s: %s", key, exc)
                    cleaned[key] = str(value)
        elif isinstance(value, dict):
            try:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                logger.debug("Failed to JSON-encode %s: %s", key, exc)
                cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)

    return cleaned
