"""The LoCoMo-specific decisions in the LangGraph agent.

Rendering used to live here too and is now
:func:`graphknows.models.hit.render_memories`, covered in
``tests/models/test_hit.py`` — including the one behaviour both arms depend on:
a hit with a ``ts`` renders dated and attributed, a hit without one (the
``--prefix`` arm) passes through carrying its own inline date.

What stays LoCoMo's own is how a dataset message becomes a turn.
"""

from __future__ import annotations

from evaluation.scripts.locomo_langgraph import parse_turn


def test_parse_turn_routes_each_field_to_its_own_home() -> None:
    """The speaker and timestamp are metadata, deliberately not text.

    Flattening them into the utterance is what made the extractor re-derive
    them as entities: on conv-30 "Jon" never existed as a node, but "Hey Jon",
    "Thanks, Jon!" and "Oh no, Jon!" did, across 15 fragments.
    """
    msg = {"role": "Gina", "timestamp": "4:04 pm on 20 January, 2023", "content": "Hey Jon!"}

    assert parse_turn(msg) == {
        "speaker": "Gina",
        "timestamp": "4:04 pm on 20 January, 2023",
        "text": "Hey Jon!",
    }


def test_prefix_flag_reproduces_the_baseline_exactly() -> None:
    """Full replication: the text is flattened AND the metadata fields are left
    empty, so CHUNK.speaker falls back to the role and CHUNK.ts stays "". A
    prefix that also set the fields would be a third design, testing nothing."""
    msg = {"role": "Gina", "timestamp": "4:04 pm on 20 January, 2023", "content": "Hey Jon!"}

    assert parse_turn(msg, prefix=True) == {
        "speaker": "",
        "timestamp": "",
        "text": "4:04 pm on 20 January, 2023 | Gina: Hey Jon!",
    }
