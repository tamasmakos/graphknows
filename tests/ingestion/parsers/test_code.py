"""The code parser: one segment per definition, cited by qualified symbol."""

from __future__ import annotations

from graphknows.ingestion.parsers.code import PythonCodeParser
from graphknows.ingestion.parsers.registry import Parser
from graphknows.models.segment import SegmentKind
from graphknows.models.source import Source

SOURCE = Source(uri="pkg/store.py", content_hash="deadbeef", mime="text/x-python")

CODE = b'''"""Module docstring."""

CONSTANT = 1


def put(key):
    return key


class Store:
    def get(self, key):
        def _inner():
            return key

        return _inner()

    async def flush(self):
        pass
'''


def _segments():
    return list(PythonCodeParser().parse(SOURCE, CODE))


def test_the_parser_conforms_to_the_parser_seam():
    assert isinstance(PythonCodeParser(), Parser)


def test_every_function_and_class_becomes_one_segment_by_qualified_symbol():
    assert [s.path for s in _segments()] == [
        "put",
        "Store",
        "Store.get",
        "Store.get._inner",
        "Store.flush",
    ]


def test_a_segment_is_the_definition_it_cites():
    assert {s.kind for s in _segments()} == {SegmentKind.code}

    flush = next(s for s in _segments() if s.path == "Store.flush")
    assert flush.text.startswith("async def flush")
    assert CODE[slice(*flush.byte_range)].decode() == flush.text
    assert flush.source_id == SOURCE.id
    assert flush.extractor_version == PythonCodeParser.version
