"""Entity resolution for STM->LTM promotion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from graphknows.memory.models.contracts import PromotionAudit


class EntityResolution:
    """Resolve extracted STM entities into canonical LTM entities."""

    def resolve(
        self,
        entities: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], PromotionAudit]:
        """Return canonical entities and an audit of merged aliases.

        The first implementation preserves current behavior: entities are
        grouped case-insensitively and the highest-confidence spelling wins.
        The audit makes that decision visible for later, richer resolution.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if name:
                groups[name.casefold()].append(entity)

        resolved: list[dict[str, Any]] = []
        aliases: dict[str, list[str]] = {}
        merge_count = 0

        for group in groups.values():
            best = max(group, key=lambda item: float(item.get("score", item.get("confidence", 1.0)) or 0.0))
            canonical_name = str(best.get("name", "")).strip()
            alias_names = sorted(
                {
                    str(item.get("name", "")).strip()
                    for item in group
                    if str(item.get("name", "")).strip() != canonical_name
                }
            )
            if len(group) > 1:
                merge_count += len(group) - 1
                aliases[canonical_name] = alias_names
            resolved.append({**best, "aliases": alias_names})

        return resolved, PromotionAudit(entity_aliases=aliases, merged_entities=merge_count)
