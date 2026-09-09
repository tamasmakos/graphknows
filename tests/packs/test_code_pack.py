"""The code pack: its vocabulary is data, its facts are syntax, its veto is FR-019."""

from __future__ import annotations

from graphknows.ingestion.extraction.protocol import Extractor, PackGuidance
from graphknows.models.segment import Segment, SegmentKind
from graphknows.packs import DomainPack, load_packs
from graphknows.packs.code import CodeExtractor, CodePack, symbol_id

_FUNCTION = """def load(path):
    import json

    return json.loads(read(path))
"""

_TEST = """def test_load():
    assert load("x")
"""

_CLASS = """class Loader:
    def load(self):
        return None
"""

_MODULE = """import json


def load(path):
    return json.loads(path)
"""


def _segment(text: str, path: str) -> Segment:
    return Segment(source_id="src", text=text, kind=SegmentKind.code, path=path)


def _triples(text: str, path: str) -> set[tuple[str, str, str]]:
    """The facts the extractor reads off *text*, as (subject, predicate, object)."""
    _, facts = CodeExtractor().extract(_segment(text, path), CodePack())
    return {(fact.subject, fact.predicate, fact.object) for fact in facts}


def test_the_pack_satisfies_both_seams() -> None:
    assert isinstance(CodePack(), DomainPack)
    assert isinstance(CodePack(), PackGuidance)
    assert isinstance(CodeExtractor(), Extractor)


def test_the_vocabulary_comes_from_the_data_file() -> None:
    loaded = load_packs([CodePack()])

    assert set(loaded.concepts) == {"code:function", "code:class", "code:module", "code:file"}
    assert set(loaded.predicates) == {
        "code:calls",
        "code:imports",
        "code:defines",
        "code:tests",
    }
    assert {concept.pack for concept in loaded.concepts.values()} == {"code"}
    assert loaded.predicates["code:calls"].canonical == "calls"


def test_entity_labels_are_the_concept_labels() -> None:
    assert CodePack().entity_labels() == ("function", "class", "module", "file")


def test_hygiene_admits_symbols_and_refuses_prose() -> None:
    admits = CodePack().hygiene()

    assert admits("load") and admits("graphknows.packs.code")
    assert not admits("the loader we discussed")


def test_a_function_and_a_module_of_one_name_never_merge() -> None:
    pack = CodePack()

    assert pack.veto(symbol_id("function", "code"), symbol_id("module", "code"))
    assert pack.veto(symbol_id("module", "Code"), symbol_id("function", "code"))


def test_the_veto_is_silent_about_everything_else() -> None:
    pack = CodePack()

    assert not pack.veto(symbol_id("function", "load"), symbol_id("function", "load"))
    assert not pack.veto(symbol_id("function", "load"), symbol_id("module", "loader"))
    assert not pack.veto(symbol_id("function", "load"), "deadbeefdeadbeef")


def test_a_function_segment_asserts_its_calls_and_imports() -> None:
    load = symbol_id("function", "load")

    assert _triples(_FUNCTION, "loader.load") == {
        (load, "code:calls", symbol_id("function", "loads")),
        (load, "code:calls", symbol_id("function", "read")),
        (load, "code:imports", symbol_id("module", "json")),
    }


def test_a_test_function_tests_what_it_calls() -> None:
    assert _triples(_TEST, "test_loader.test_load") == {
        (
            symbol_id("function", "test_load"),
            "code:tests",
            symbol_id("function", "load"),
        )
    }


def test_a_class_segment_defines_its_methods() -> None:
    assert _triples(_CLASS, "loader.Loader") == {
        (symbol_id("class", "Loader"), "code:defines", symbol_id("function", "load"))
    }


def test_a_module_segment_is_the_subject_of_its_own_statements() -> None:
    module = symbol_id("module", "loader")

    assert _triples(_MODULE, "loader") == {
        (module, "code:defines", symbol_id("function", "load")),
        (module, "code:imports", symbol_id("module", "json")),
        (module, "code:calls", symbol_id("function", "loads")),
    }


def test_mentions_cite_bytes_of_the_segment() -> None:
    segment = _segment(_FUNCTION, "loader.load")

    mentions, _ = CodeExtractor().extract(segment, CodePack())

    by_id = {mention.entity_id: mention for mention in mentions}
    read = by_id[symbol_id("function", "read")]
    start, end = read.span
    assert _FUNCTION.encode()[start:end].decode() == "read"
    assert read.label == "function"
    assert read.segment_id == segment.id


def test_the_extractor_is_deterministic_across_runs() -> None:
    first = CodeExtractor().extract(_segment(_FUNCTION, "loader.load"), CodePack())
    second = CodeExtractor().extract(_segment(_FUNCTION, "loader.load"), CodePack())

    assert first == second


def test_a_segment_that_is_not_parseable_code_says_nothing() -> None:
    assert _triples("def load(:", "loader.load") == set()

    prose = Segment(source_id="src", text=_FUNCTION, kind=SegmentKind.prose, path="loader.load")
    assert CodeExtractor().extract(prose, CodePack()) == ([], [])
