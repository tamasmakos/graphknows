"""Evidence-recall: the un-inflatable retrieval metric."""

from __future__ import annotations

from evaluation.common.lexical import evidence_recall
from evaluation.locomo.dataset import _evidence_turns

ROW = {
    "evidence": ["D1:3", "D2:1"],
    "sessions": [
        {"messages": [{"content": "one"}, {"content": "two"}, {"content": "I went camping"}]},
        {"messages": [{"content": "we moved to Seattle"}]},
    ],
}


class TestEvidenceTurns:
    def test_resolves_one_indexed_doc_and_turn(self) -> None:
        assert _evidence_turns(ROW) == ["I went camping", "we moved to Seattle"]

    def test_handles_stringified_evidence_lists(self) -> None:
        assert _evidence_turns({**ROW, "evidence": "['D1:3']"}) == ["I went camping"]

    def test_missing_or_out_of_range_evidence_is_skipped(self) -> None:
        assert _evidence_turns({"sessions": ROW["sessions"]}) == []
        assert _evidence_turns({**ROW, "evidence": ["D9:9"]}) == []


class TestEvidenceRecall:
    def test_per_hop_fraction(self) -> None:
        ev = ["I went camping", "we moved to Seattle"]
        assert evidence_recall(ev, ["... I went camping ..."]) == 0.5
        assert evidence_recall(ev, ["I went camping", "we moved to Seattle"]) == 1.0
        assert evidence_recall(ev, ["unrelated"]) == 0.0

    def test_whitespace_normalised(self) -> None:
        assert evidence_recall(["I went camping"], ["x\n I  went\ncamping \n y"]) == 1.0

    def test_padding_the_haystack_cannot_inflate_it(self) -> None:
        """The property gold_reachable lacks: irrelevant passages add nothing."""
        ev = ["I went camping", "we moved to Seattle"]
        noise = [f"irrelevant passage {i}" for i in range(50)]
        assert evidence_recall(ev, ["I went camping", *noise]) == 0.5

    def test_no_evidence_is_zero_and_callers_skip_it(self) -> None:
        assert evidence_recall([], ["anything"]) == 0.0
