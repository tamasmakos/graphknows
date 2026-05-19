"""
ConnectorRegistry — loads source_manifest.yaml and instantiates connectors.

Usage
-----
    registry = ConnectorRegistry()
    connector = registry.get("eurlex")
    all_ids   = registry.list_ids()
    summary   = registry.summary()          # agent-readable string
    entries   = registry.list_entries()     # raw manifest dicts
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .connectors.base import SourceConnector

log = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).with_name("source_manifest.yaml")


@dataclass
class SourceEntry:
    """Parsed representation of one source_manifest.yaml entry."""

    id: str
    name: str
    description: str
    connector_class: str
    jurisdictions: list[str]
    doc_types: list[str]
    temporal_coverage: str
    query_strengths: list[str]
    language: str = "en"
    rate_limit_rps: float = 1.0
    requires_auth: bool = False
    enabled: bool = True


class ConnectorRegistry:
    """Loads source_manifest.yaml and provides named access to connectors.

    Connectors are instantiated lazily on first access and cached.
    """

    def __init__(self, manifest_path: Path = _MANIFEST_PATH) -> None:
        self._manifest_path = manifest_path
        self._entries: dict[str, SourceEntry] = {}
        self._instances: dict[str, SourceConnector] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, source_id: str) -> Optional[SourceConnector]:
        """Return the connector instance for *source_id*, or None."""
        entry = self._entries.get(source_id)
        if entry is None or not entry.enabled:
            return None
        if source_id not in self._instances:
            self._instances[source_id] = self._instantiate(entry)
        return self._instances[source_id]

    def list_ids(self, jurisdiction: str | None = None, doc_type: str | None = None) -> list[str]:
        """Return enabled source IDs, optionally filtered."""
        return [
            e.id
            for e in self._entries.values()
            if e.enabled
            and (jurisdiction is None or jurisdiction in e.jurisdictions)
            and (doc_type is None or doc_type in e.doc_types)
        ]

    def list_entries(
        self,
        jurisdiction: str | None = None,
        doc_type: str | None = None,
    ) -> list[SourceEntry]:
        """Return SourceEntry objects for all enabled (and optionally filtered) sources."""
        return [
            e
            for e in self._entries.values()
            if e.enabled
            and (jurisdiction is None or jurisdiction in e.jurisdictions)
            and (doc_type is None or doc_type in e.doc_types)
        ]

    def build(self) -> dict[str, SourceConnector]:
        """Return a dict of {source_id: connector} for all enabled sources."""
        result: dict[str, SourceConnector] = {}
        for entry in self._entries.values():
            if not entry.enabled:
                continue
            connector = self.get(entry.id)
            if connector is not None:
                result[entry.id] = connector
        return result

    def summary(self) -> str:
        """Return an agent-readable summary of all available sources."""
        lines: list[str] = ["Available EU Parliamentary Data Sources:", ""]
        for entry in self._entries.values():
            if not entry.enabled:
                continue
            strengths = ", ".join(entry.query_strengths[:3])
            lines.append(
                f"[{entry.id}] {entry.name}\n"
                f"  {entry.description.strip()[:180]}\n"
                f"  Coverage: {entry.temporal_coverage} | "
                f"Lang: {entry.language} | "
                f"Jurisdictions: {', '.join(entry.jurisdictions)}\n"
                f"  Best for: {strengths}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        try:
            raw = yaml.safe_load(self._manifest_path.read_text())
        except Exception as exc:
            log.error("Failed to load source_manifest.yaml: %s", exc)
            return
        for item in raw.get("sources", []):
            try:
                entry = SourceEntry(
                    id=item["id"],
                    name=item["name"],
                    description=item.get("description", ""),
                    connector_class=item["connector_class"],
                    jurisdictions=item.get("jurisdictions", ["EU"]),
                    doc_types=item.get("doc_types", []),
                    temporal_coverage=item.get("temporal_coverage", ""),
                    query_strengths=item.get("query_strengths", []),
                    language=item.get("language", "en"),
                    rate_limit_rps=float(item.get("rate_limit_rps", 1.0)),
                    requires_auth=bool(item.get("requires_auth", False)),
                    enabled=bool(item.get("enabled", True)),
                )
                self._entries[entry.id] = entry
            except Exception as exc:
                log.warning("Skipping malformed manifest entry: %s — %s", item.get("id"), exc)

    def _instantiate(self, entry: SourceEntry) -> Optional[SourceConnector]:
        module_path, class_name = entry.connector_class.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            log.warning("Connector %s unavailable — missing dependency: %s", entry.id, exc)
            return None
        cls = getattr(module, class_name)
        return cls()


# Module-level singleton for convenience
_registry: Optional[ConnectorRegistry] = None


def get_registry() -> ConnectorRegistry:
    """Return the module-level ConnectorRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry
