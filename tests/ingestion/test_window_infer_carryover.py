"""A window extracts when any turn folded into it asked to be extracted.

With window chunking on, a turn does not get its own chunk: it is folded into
the window that a later turn emits. The emitted chunk carries the folded turns'
text, so it has to carry their extract/skip decision too. Taking only the
emitting turn's flag meant one trailing ``infer=False`` turn -- tool output, a
system message, a greeting -- silenced extraction for everything said before it,
and the whole window's entities were never minted.

Guarded here rather than only in ``tests/memory/test_turn_buffer.py``: that one
needs a live ArcadeDB, which is why this went unseen.
"""

from __future__ import annotations

from graphknows.ingestion.stm.ingest import _STMIngestHandler


def _handler(stride: int = 3) -> _STMIngestHandler:
    """A handler with only the window accumulator wired up.

    ``_chunk_stride`` is a read-only property (window minus overlap), so the
    two values it derives from are what get set.
    """
    handler = _STMIngestHandler.__new__(_STMIngestHandler)  # no store/extractor needed
    handler._window_chunks = True
    handler._extract_window = stride
    handler._window_overlap = 0
    handler._since_emit = {}
    handler._pending_infer = {}
    assert handler._chunk_stride == stride, "stride is window minus overlap"
    return handler


def test_an_earlier_infer_turn_survives_a_trailing_skip() -> None:
    """The bug: Gina's turn asked to be extracted, Jon's greeting emitted it."""
    handler = _handler()
    assert handler._window_decision("s1", infer=True, last_in_batch=False) == (False, True)
    assert handler._window_decision("s1", infer=False, last_in_batch=True) == (True, True)


def test_a_window_of_skipped_turns_still_mints_nothing() -> None:
    """The carry-over must not turn every window into an extraction."""
    handler = _handler()
    handler._window_decision("s1", infer=False, last_in_batch=False)
    assert handler._window_decision("s1", infer=False, last_in_batch=True) == (True, False)


def test_the_pending_flag_does_not_leak_into_the_next_window() -> None:
    """One window's decision is its own; the next starts from its own turns."""
    handler = _handler()
    handler._window_decision("s1", infer=True, last_in_batch=False)
    assert handler._window_decision("s1", infer=False, last_in_batch=True) == (True, True)
    assert handler._window_decision("s1", infer=False, last_in_batch=True) == (True, False)


def test_sessions_do_not_share_a_pending_decision() -> None:
    """Two conversations ingesting concurrently keep their own flags."""
    handler = _handler()
    handler._window_decision("s1", infer=True, last_in_batch=False)
    assert handler._window_decision("s2", infer=False, last_in_batch=True) == (True, False)
    assert handler._window_decision("s1", infer=False, last_in_batch=True) == (True, True)


def test_a_full_stride_emits_without_waiting_for_the_batch_end() -> None:
    """Windowing is a stride, not a batch boundary: the nth turn emits."""
    handler = _handler(stride=2)
    assert handler._window_decision("s1", infer=True, last_in_batch=False) == (False, True)
    assert handler._window_decision("s1", infer=False, last_in_batch=False) == (True, True)
