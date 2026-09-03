"""Transformer frame identification: offsets, trigger marking, instance arity.

The parser itself is an optional extra and a seq2seq model, so it is never
loaded here — these cover the pure text handling plus the one behavioural claim
that distinguishes the SRL path from the default one: a chunk yields an instance
per trigger instead of at most one.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from graphknows.ingestion.extraction.relations.frame_srl import extract_frame_instances_srl
from graphknows.symbolic.framenet.srl import marked_line, sentences, srl_enabled

CHUNK = "Caroline: I went to a support group.\nMelanie: I gave a book and gave a pen."


def test_sentences_carry_offsets_into_the_chunk() -> None:
    for offset, sentence in sentences(CHUNK):
        assert CHUNK[offset : offset + len(sentence)] == sentence


def test_turn_without_final_punctuation_still_splits_from_the_next_speaker() -> None:
    # A chat turn need not end in .!? — without the newline split, two speakers
    # would land in one "sentence" and the second speaker's roles would resolve
    # to the first.
    text = "Caroline: hey\nMelanie: I painted a lake sunrise."
    assert [s for _, s in sentences(text)] == [
        "Caroline: hey",
        "Melanie: I painted a lake sunrise.",
    ]


def test_marked_line_keeps_the_speaker_prefix() -> None:
    offset = CHUNK.index("went")
    assert marked_line(CHUNK, offset, "went") == "Caroline: I *went* to a support group."


def test_same_frame_twice_on_one_line_gets_distinct_inputs() -> None:
    first, second = CHUNK.index("gave"), CHUNK.rindex("gave")
    assert marked_line(CHUNK, first, "gave") != marked_line(CHUNK, second, "gave")


def test_marked_line_scopes_to_the_sentence_not_the_whole_turn() -> None:
    """marked_line's return value is the ONLY context GLiNER2 sees when filling
    a frame's core roles (see extract_frame_instances_srl). A chat turn is one
    LINE but can hold several sentences about several events, so a line-wide
    context lets a Firing instance's role filler collect entities from the same
    turn's Hiring sentence, and vice versa (issue #161's cross-event bleed).
    Scoping to the trigger's SENTENCE fixes that; the ``Speaker:`` prefix must
    still survive so a 1st-person filler ("I") can resolve to the speaker.
    """
    text = "Caroline: Tom got fired from the factory. Caroline got hired at the hospital."
    result = marked_line(text, text.index("fired"), "fired")
    assert "*fired*" in result
    assert "factory" in result
    assert "hospital" not in result
    assert "hired" not in result
    assert result.startswith("Caroline:")


def test_marked_line_prefixes_speaker_for_a_trigger_in_the_second_sentence() -> None:
    """The speaker prefix must be reattached even when the trigger falls in a
    later sentence of the turn, since narrowing to the sentence would otherwise
    drop the turn's ``Speaker:`` prefix along with the other sentence's text.
    """
    text = "Caroline: Tom got fired from the factory. Caroline got hired at the hospital."
    result = marked_line(text, text.index("hired"), "hired")
    assert "*hired*" in result
    assert result.startswith("Caroline:")
    assert "factory" not in result
    assert "fired" not in result


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("fst", True), ("false", False), ("1", True), ("0", False)],
)
def test_the_env_var_overrides_either_way(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("GRAPHKNOWS_FRAME_SRL", value)
    assert srl_enabled() is expected


def test_installed_is_false_when_the_spec_exists_but_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A broken/partial install: find_spec sees a spec, but the real import
    # still fails -- a missing transitive dependency, say. installed() must not
    # trust find_spec alone, or ingest crashes on MissingExtraError instead of
    # degrading to the LU identifier.
    from graphknows.symbolic.framenet import srl

    srl.installed.cache_clear()
    monkeypatch.setattr(srl.importlib.util, "find_spec", lambda name: object())
    # None in sys.modules makes `import frame_semantic_transformer` raise
    # ImportError, so this holds whether or not the extra is really installed.
    monkeypatch.setitem(sys.modules, "frame_semantic_transformer", None)
    try:
        assert srl.installed() is False
    finally:
        srl.installed.cache_clear()


def test_unset_follows_whether_the_extra_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default ON where the parser exists (measured: conv-30 0.778 -> 0.889), and
    # a base install without the extra keeps the LU identifier rather than
    # raising on every ingest.
    monkeypatch.delenv("GRAPHKNOWS_FRAME_SRL", raising=False)
    from graphknows.symbolic.framenet import srl

    monkeypatch.setattr(srl, "installed", lambda: True)
    assert srl.srl_enabled() is True
    monkeypatch.setattr(srl, "installed", lambda: False)
    assert srl.srl_enabled() is False


class _Gliner:
    """Fills two core roles, confidently — enough to clear the fillability gate."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def inference(self, **kwargs: object) -> list[list[dict[str, object]]]:
        texts = kwargs["texts"]
        assert isinstance(texts, list)
        self.texts.append(str(texts[0]))
        labels = kwargs["labels"]
        assert isinstance(labels, list)
        return [
            [
                {"text": "Melanie", "label": str(labels[0]), "score": 0.9},
                {"text": "a book", "label": str(labels[1]), "score": 0.8},
            ]
        ]


def test_every_trigger_becomes_its_own_instance() -> None:
    hits = [
        SimpleNamespace(frame="Giving", trigger="gave", offset=CHUNK.index("gave")),
        SimpleNamespace(frame="Giving", trigger="gave", offset=CHUNK.rindex("gave")),
    ]
    gliner = _Gliner()

    instances = extract_frame_instances_srl(CHUNK, "Melanie", ["Caroline", "Melanie"], gliner, hits)

    # Two triggers -> two instances. The default path returns at most one.
    assert len(instances) == 2
    # ...and each was filled from its OWN marked line, not the shared chunk.
    assert gliner.texts[0] != gliner.texts[1]
    assert all(t.count("*") == 2 for t in gliner.texts)


def test_instances_failing_the_fillability_gate_are_dropped() -> None:
    class _Weak:
        def inference(self, **kwargs: object) -> list[list[dict[str, object]]]:
            labels = kwargs["labels"]
            assert isinstance(labels, list)
            return [[{"text": "Melanie", "label": str(labels[0]), "score": 0.1}]]

    hits = [SimpleNamespace(frame="Giving", trigger="gave", offset=CHUNK.index("gave"))]
    assert extract_frame_instances_srl(CHUNK, "Melanie", [], _Weak(), hits) == []
