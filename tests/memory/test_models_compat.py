"""Verify compat shims work."""

from graphknows.memory.models import IngestResult, RecallResult


def test_ingest_result_importable_from_models():
    r = IngestResult(session_id="s", file_id="f", title="t")
    assert r.session_id == "s"


def test_recall_result_importable_from_models():
    r = RecallResult(text="x", source_type="chunk", scope="short_term")
    assert r.text == "x"
