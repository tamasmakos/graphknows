"""FR-012: the LM builder disables dspy's on-disk cache.

dspy's default writes every prompt — ingested chunk text included — and every
completion to ``~/.dspy_cache``, outside the configured database. The
clean-``HOME`` confirmation of SC-010 needs a funded API key and stays a manual
step in quickstart.md §4; this is the part that can be checked offline.
"""

from __future__ import annotations

from graphknows.llm import build_lm


def test_built_lm_has_the_disk_cache_off() -> None:
    assert build_lm("openai/gpt-4o-mini").cache is False


def test_an_explicit_cache_argument_still_wins() -> None:
    assert build_lm("openai/gpt-4o-mini", cache=True).cache is True
