"""A grounding's ``version`` must be the ontology's CONTENT identity.

``_STMIngestHandler._link_ontology`` reads ``self._ontology.source`` for the
provenance block's ``version`` field — the ontology's FILE PATH. Two loads of
the same ontology file always carry the same path, so upgrading the ontology
in place (same path, new content) writes an identical ``version`` on every
grounding: old and new groundings become indistinguishable, and no version
can ever "coexist" per issue #145.

``OntologyIndex`` already computes ``digest``, a content hash over every
term's uri/label/definition (see ``symbolic/ontology/catalog.py``,
``_source_digest``). ``version`` must be that digest, not ``source``.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.symbolic.ontology import OntologyTerm


class _StubIndex:
    """The attributes ``_STMIngestHandler`` reads off an OntologyIndex.

    ``source`` and ``digest`` are deliberately distinct strings: ``source`` is
    the loader's file path, ``digest`` is the content hash. If ``_link_ontology``
    ever falls back to ``source`` again, the two values being different is
    exactly what makes that visible.
    """

    def __init__(self, classes: list[OntologyTerm]) -> None:
        self.classes = classes
        self.properties: list[OntologyTerm] = []
        self.source = "/opt/ontologies/cco.owl"
        self.digest = "a1b2c3d4e5f6"

    @property
    def property_labels(self) -> list[str]:
        return [t.label for t in self.properties]


class _FakeStore:
    """Records the INSTANCE_OF mappings instead of writing them."""

    def __init__(self) -> None:
        self.instance_of: list[dict[str, Any]] = []

    async def write_class_matches(self, chunk_id: str, matches: Any) -> None:
        return None

    async def write_groundings(self, rows: list[dict[str, Any]]) -> None:
        self.instance_of.extend(rows)


def _handler(store: _FakeStore, index: _StubIndex) -> _STMIngestHandler:
    return _STMIngestHandler(
        store=store, extractor=None, embed_fn=None, ontology_index=index, relation_index=index
    )


@pytest.mark.asyncio
async def test_grounding_version_is_the_ontology_digest_not_its_path() -> None:
    index = _StubIndex([OntologyTerm(uri="u:Person", label="Person", definition="", kind="class")])
    store = _FakeStore()

    await _handler(store, index)._link_ontology(
        "c1",
        [{"name": "Ada", "type": "Person", "score": 0.95}],
        [],
    )

    assert store.instance_of, "expected at least one grounding to be produced"
    versions = {m["version"] for m in store.instance_of}
    assert versions == {index.digest}, (
        f"grounding version must be the ontology's content digest {index.digest!r}, "
        f"got {versions!r} (index.source={index.source!r})"
    )
