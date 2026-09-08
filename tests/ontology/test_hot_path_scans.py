"""Three per-chunk hot-path scans that ran again on every chunk instead of once.

FR-038: ``ancestors()`` fell back to an O(n) walk over the whole CCO scheme
(~1,660 concepts) on every case-mismatched lookup -- nearly every chunk, since
the extractor's coarse types are lowercase and the scheme is keyed by
capitalised prefLabels; ``resolve_temporal`` built a fresh ``DateDataParser``
per chunk instead of once per anchor; the schema miner re-parsed the chunk
spaCy already parsed for lexical labelling / frame-candidate ranking /
modality, doubling the parse per turn. Each fix keeps behaviour identical and
only changes how many times the expensive step runs.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from graphknows.ingestion.extraction.entities.extractor import (
    GLiNER2EntityExtractor,
    _SpacySchemaMiner,
)
from graphknows.symbolic.ontology.skos import (
    Concept,
    _casefold_index,
    _casefold_index_cache,
    ancestors,
)
from graphknows.temporal import _parser_for, resolve_temporal


def _scheme(parents: dict[str, tuple[str, ...]]) -> dict[str, Concept]:
    return {
        label: Concept(
            uri=f"http://example.org/{label.replace(' ', '_')}",
            pref_label=label,
            definition="",
            kind="class",
            broader=broader,
        )
        for label, broader in parents.items()
    }


class TestAncestorsCasefoldIndex:
    """``ancestors()`` widens a case-mismatched label via a cached index, not a scan."""

    def test_case_mismatched_lookup_still_resolves(self) -> None:
        scheme = _scheme({"Person": ("Agent",), "Agent": ()})
        assert ancestors(scheme, "person") == ancestors(scheme, "Person") == ["Agent"]

    def test_index_is_built_once_per_scheme_and_reused(self) -> None:
        scheme = _scheme({"Person": ("Agent",), "Agent": ()})
        first = _casefold_index(scheme)
        second = _casefold_index(scheme)
        assert first is second, "a second call must reuse the cached index, not rebuild it"
        assert _casefold_index_cache[id(scheme)] == (scheme, first)

    def test_two_live_schemes_keep_independent_indexes(self) -> None:
        """A cache keyed by ``id()`` must not let one scheme answer for another."""
        store = _scheme({"Store": ("Facility",), "Facility": ()})
        vehicle = _scheme({"Vehicle": ("Artifact",), "Artifact": ()})
        assert ancestors(store, "store") == ["Facility"]
        assert ancestors(vehicle, "vehicle") == ["Artifact"]


class TestTemporalParserCache:
    """``resolve_temporal`` reuses one ``DateDataParser`` per anchor instead of per chunk."""

    def test_parser_is_shared_across_chunks_with_the_same_anchor(self) -> None:
        anchor = datetime(2023, 5, 25)
        first = _parser_for(anchor)
        second = _parser_for(anchor)
        assert first is second, "chunks sharing an anchor must reuse the same parser"

    def test_distinct_anchors_get_distinct_but_correct_parsers(self) -> None:
        early = _parser_for(datetime(2020, 1, 1))
        late = _parser_for(datetime(2023, 5, 25))
        assert early is not late

    def test_resolve_temporal_still_resolves_relative_dates_after_reuse(self) -> None:
        anchor = datetime(2023, 5, 25)
        ents = [{"name": "two weeks ago", "type": "DATE"}]
        # Two calls with the same anchor exercise the reused (not rebuilt) parser.
        resolve_temporal(ents, anchor)
        out = {o["raw"]: o["iso"] for o in resolve_temporal(ents, anchor)}
        assert out["two weeks ago"] == "2023-05-11"


class TestSchemaMinerDocReuse:
    """The chunk is parsed once per turn, not once for schema mining and again for extract()."""

    def test_build_accepts_a_preparsed_doc_and_never_reparses(self) -> None:
        text = "Caroline: I work as a nurse at Mercy Hospital."
        baseline = _SpacySchemaMiner().build(text)
        doc = _SpacySchemaMiner().nlp(text)

        class _ExplodingMiner(_SpacySchemaMiner):
            @property
            def nlp(self) -> object:  # type: ignore[override]
                raise AssertionError("build() must not reparse when a doc is supplied")

        result = _ExplodingMiner().build(text, doc=doc)
        assert result.entity_spec == baseline.entity_spec

    def test_extract_parses_the_chunk_exactly_once(self, monkeypatch) -> None:
        text = "Gina: I opened an online clothing store."
        calls: list[str] = []
        real_pipeline = _SpacySchemaMiner().nlp

        class _CountingPipeline:
            def __call__(self, parsed_text: str) -> object:
                calls.append(parsed_text)
                return real_pipeline(parsed_text)

            def pipe(self, texts: object) -> object:
                return real_pipeline.pipe(texts)

        monkeypatch.setattr(_SpacySchemaMiner, "nlp", property(lambda self: _CountingPipeline()))

        mock_model = MagicMock()
        mock_model.inference.return_value = (
            [[{"text": "Gina", "label": "person", "score": 0.9}]],
            [[]],
        )
        extractor = GLiNER2EntityExtractor(
            model_loader=lambda: mock_model, schema_miner=_SpacySchemaMiner()
        )

        extractor.extract(text)

        assert calls == [text], "the window must be parsed once, not once per caller"
