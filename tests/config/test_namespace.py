"""Tests for namespace → database-name resolution."""

from __future__ import annotations

from graphknows.storage.namespace import sanitize


class TestSanitize:
    def test_empty_and_blank(self) -> None:
        assert sanitize("") == ""
        assert sanitize("   ") == ""

    def test_lowercases_and_replaces_invalid(self) -> None:
        assert sanitize("Weird/Name@#") == "weird_name"
        assert sanitize("ACME Corp!") == "acme_corp"

    def test_keeps_valid_chars(self) -> None:
        assert sanitize("eval_locomo") == "eval_locomo"
        assert sanitize("tenant_42") == "tenant_42"

    def test_overlong_is_hashed_but_distinct(self) -> None:
        a = sanitize("x" * 200)
        b = sanitize("y" * 200)
        assert len(a) <= 48 and len(b) <= 48
        assert a != b  # distinct inputs stay distinct
