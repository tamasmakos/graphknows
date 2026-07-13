"""Tests for namespace → database-name resolution."""

from __future__ import annotations

from graphknows.config.namespace import db_names, namespaces_from_databases, sanitize


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


class TestDbNames:
    def test_default_namespace_is_backward_compatible(self) -> None:
        assert db_names("") == ("stm", "ltm")
        assert db_names("   ") == ("stm", "ltm")

    def test_namespaced(self) -> None:
        assert db_names("acme") == ("stm_acme", "ltm_acme")
        assert db_names("eval_locomo") == ("stm_eval_locomo", "ltm_eval_locomo")

    def test_sanitizes_before_naming(self) -> None:
        assert db_names("ACME Corp!") == ("stm_acme_corp", "ltm_acme_corp")


class TestNamespacesFromDatabases:
    def test_empty(self) -> None:
        assert namespaces_from_databases([]) == [""]

    def test_default_pair_yields_default_namespace(self) -> None:
        assert namespaces_from_databases(["stm", "ltm"]) == [""]

    def test_derives_and_sorts_with_default_first(self) -> None:
        dbs = ["ltm", "stm_acme", "ltm_acme", "ltm_beta", "unrelated"]
        assert namespaces_from_databases(dbs) == ["", "acme", "beta"]

    def test_single_half_of_a_pair_still_counts(self) -> None:
        # Only the LTM half exists — the namespace is still reported.
        assert namespaces_from_databases(["ltm_itest_a"]) == ["", "itest_a"]

    def test_ignores_bare_prefixes(self) -> None:
        # "stm_"/"ltm_" with nothing after the underscore is not a namespace.
        assert namespaces_from_databases(["stm_", "ltm_"]) == [""]
