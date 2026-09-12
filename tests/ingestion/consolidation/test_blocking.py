"""Blocking partitions entities coarsely enough to catch spelling drift (FR-020)."""

from __future__ import annotations

import pytest

from graphknows.ingestion.consolidation.blocking import block_key, dominant_type

pytestmark = pytest.mark.unit


def test_spelling_drift_shares_a_block() -> None:
    assert block_key("acme corp", {"ORG": 1}) == block_key("acme corporation", {"ORG": 2})


def test_unrelated_names_are_separate_blocks() -> None:
    assert block_key("acme corp", {"ORG": 1}) != block_key("globex", {"ORG": 1})


def test_type_separates_a_shared_prefix() -> None:
    assert block_key("mercury", {"PLANET": 1}) != block_key("mercury", {"ELEMENT": 1})


def test_untyped_entities_block_on_the_prefix_alone() -> None:
    assert block_key("acme corp", {}) == block_key("acme air", {})


def test_short_names_keep_their_whole_key() -> None:
    assert block_key("ab", {}) == "ab|"


def test_dominant_type_is_the_most_observed_label() -> None:
    assert dominant_type({"ORG": 1, "PERSON": 3}) == "PERSON"


def test_dominant_type_breaks_ties_alphabetically() -> None:
    assert dominant_type({"PERSON": 2, "ORG": 2}) == "ORG"
