"""Guard tests for GLiNER2 relation-object cleaning (object_cleaner.py).

Uses the real spaCy pipeline (as the ingest path does) to assert the four
structural drop/trim rules and that clean noun objects pass through unchanged.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.extraction.relations.llm_free import clean_relation_object
from graphknows.ingestion.extraction.entities.extractor import _SpacySchemaMiner

_REL_TERMS = frozenset({"husband", "wife", "best friend"})


@pytest.fixture(scope="module")
def nlp():
    return _SpacySchemaMiner().nlp


def _clean(nlp, tail):
    return clean_relation_object(tail, nlp, _REL_TERMS)


def test_clean_noun_objects_pass_through(nlp):
    for obj in ["camping trip", "LGBTQ activist group", "school event"]:
        assert _clean(nlp, obj) == obj


def test_question_echo_dropped(nlp):
    # The named junk class: `plans_to "Any plans coming up"`.
    assert _clean(nlp, "Any plans coming up") is None
    assert _clean(nlp, "what movie") is None


def test_verb_clause_trimmed_to_head_noun(nlp):
    # A verb-headed clause keeps only its content noun, not the verb wrapper.
    assert _clean(nlp, "expressing my trans experience") == "trans experience"
    assert _clean(nlp, "make a difference") == "difference"


def test_relationship_role_dropped(nlp):
    # The named junk class: `owns husband`.
    assert _clean(nlp, "husband") is None
    assert _clean(nlp, "best friend") is None


def test_single_gerund_activity_kept(nlp):
    # A single gerund used as an activity noun survives (common hobby object).
    assert _clean(nlp, "camping") == "camping"


def test_empty_dropped(nlp):
    assert _clean(nlp, "") is None
    assert _clean(nlp, "   ") is None
