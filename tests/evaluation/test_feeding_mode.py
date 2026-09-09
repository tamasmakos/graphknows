"""Turn-fed rows, and the feeding mode the panel report states for them (FR-039)."""

from __future__ import annotations

from evaluation.beam.dataset import build_documents as beam_documents
from evaluation.common.datamodels import EvalDocument, IngestUnit
from evaluation.common.pipeline import feeding_mode
from evaluation.longmem.dataset import build_documents as longmem_documents

LONGMEM_RECORD = {
    "haystack_session_ids": ["s0"],
    "haystack_dates": ["2023/05/30 (Tue) 23:40"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "I adopted a beagle."},
            {"role": "assistant", "content": "Congratulations!"},
        ]
    ],
}

BEAM_RECORD = {
    "chat": [
        [
            {"role": "user", "content": "Book me a window seat.", "time_anchor": "2024-01-02"},
            {"role": "assistant", "content": "Done."},
        ]
    ]
}


def _units(*documents: EvalDocument) -> list[IngestUnit]:
    return [IngestUnit(group_id="g", documents=list(documents))]


class TestFeedingMode:
    def test_turns_report_turn_by_turn(self) -> None:
        doc = EvalDocument(title="s", text="", turns=[{"role": "user", "content": "hi"}])
        assert feeding_mode(_units(doc)) == "turn_by_turn"

    def test_blob_reports_document(self) -> None:
        assert feeding_mode(_units(EvalDocument(title="s", text="hi"))) == "document"

    def test_no_units_reports_document(self) -> None:
        assert feeding_mode([]) == "document"


class TestLongMemTurnFeeding:
    def test_turn_fed_keeps_messages_apart(self) -> None:
        docs = longmem_documents(LONGMEM_RECORD, max_documents=5, max_chars=1000, turn_fed=True)
        assert [t["content"] for t in docs[0].turns] == [
            "I adopted a beagle.",
            "Congratulations!",
        ]
        assert docs[0].turns[0]["timestamp"] == docs[0].anchor_date
        assert feeding_mode(_units(*docs)) == "turn_by_turn"

    def test_document_fed_by_default(self) -> None:
        docs = longmem_documents(LONGMEM_RECORD, max_documents=5, max_chars=1000)
        assert docs[0].turns == []
        assert feeding_mode(_units(*docs)) == "document"


class TestBeamTurnFeeding:
    def test_turn_fed_keeps_messages_apart(self) -> None:
        docs = beam_documents(BEAM_RECORD, max_documents=5, max_chars=1000, turn_fed=True)
        assert [t["content"] for t in docs[0].turns] == ["Book me a window seat.", "Done."]
        assert docs[0].turns[0]["timestamp"] == "2024-01-02"
        assert feeding_mode(_units(*docs)) == "turn_by_turn"

    def test_document_fed_by_default(self) -> None:
        docs = beam_documents(BEAM_RECORD, max_documents=5, max_chars=1000)
        assert docs[0].turns == []
        assert feeding_mode(_units(*docs)) == "document"
