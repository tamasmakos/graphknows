"""Abstract base class for all source adapters."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from search.models import Document, SearchQuery, SearchResult


class SourceAdapter(ABC):
    source_id: str
    rate_limit: float = 1.0  # requests per second

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]: ...

    @abstractmethod
    async def fetch(self, document_id: str) -> Document: ...

    @abstractmethod
    def canonical_id(self, raw: dict) -> str: ...

    def already_ingested(self, document_id: str, state_path: str) -> bool:
        if not os.path.exists(state_path):
            return False
        with open(state_path) as f:
            state = json.load(f)
        return document_id in state.get("ingested_ids", [])
