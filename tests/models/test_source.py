"""Source identity: same bytes at the same URI are the same source."""

from __future__ import annotations

from datetime import UTC, datetime

from graphknows.models import Source


class TestSourceId:
    def test_id_is_stable_for_the_same_uri_and_content(self) -> None:
        """Re-importing an unchanged file must be idempotent."""
        first = Source(uri="file:///notes.md", content_hash="abc")
        second = Source(uri="file:///notes.md", content_hash="abc", mime="text/markdown")

        assert first.id == second.id

    def test_changed_content_at_the_same_uri_is_a_new_source(self) -> None:
        edited = Source(uri="file:///notes.md", content_hash="def")

        assert edited.id != Source(uri="file:///notes.md", content_hash="abc").id

    def test_same_content_at_a_different_uri_is_a_new_source(self) -> None:
        copy = Source(uri="file:///copy.md", content_hash="abc")

        assert copy.id != Source(uri="file:///notes.md", content_hash="abc").id


class TestSourceFields:
    def test_imported_at_defaults_to_now_in_utc(self) -> None:
        source = Source(uri="file:///notes.md", content_hash="abc")

        assert source.imported_at.tzinfo is not None
        assert (datetime.now(UTC) - source.imported_at).total_seconds() < 60

    def test_carries_namespace_mime_and_meta(self) -> None:
        source = Source(
            uri="https://example.com/paper",
            content_hash="abc",
            mime="text/html",
            namespace="research",
            meta={"title": "Tensor Brain"},
        )

        assert (source.mime, source.namespace) == ("text/html", "research")
        assert source.meta["title"] == "Tensor Brain"
