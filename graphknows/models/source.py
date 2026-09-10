"""Source: one ingested thing — a file, a session transcript, a URL."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel, frozen=True):
    """Identity and provenance of one ingested thing.

    ``id`` is derived, not supplied: the same bytes fetched from the same
    location are the same source, so re-importing is idempotent and a changed
    file at a stable URI is a new source rather than an overwrite.

    Attributes:
        uri: Where the content came from.
        content_hash: Digest of the content the caller read from ``uri``.
        mime: Media type of the content.
        imported_at: When this import happened; defaults to now (UTC).
        namespace: The namespace the source belongs to.
        meta: Free-form provenance the caller wants carried along.
    """

    uri: str
    content_hash: str
    mime: str = ""
    imported_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    namespace: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministic identity of ``uri`` at ``content_hash``."""
        return sha256(f"{self.uri}\n{self.content_hash}".encode()).hexdigest()[:16]


def session_uri(session_id: str) -> str:
    """The ``uri`` a conversation session is ingested under.

    A session is a ``Source``; this is the one place its uri is spelled, so the
    facade that writes it and the retriever that scopes recall to it agree.
    """
    return f"session:{session_id}"
