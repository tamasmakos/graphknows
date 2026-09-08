"""FR-027: one flush, one read of the entity table — and bounded work over it.

The layered resolver used to re-read the whole (unbounded) ENTITY table and the
speaker list once per layer, so that a later layer could see the earlier ones'
effects. It now reads both once into ``_Namespace`` and prunes that set in
place. These tests pin the three costs the requirement names:

* the table is fetched exactly once per flush (and the shrinking set still
  behaves as if every layer had re-read it),
* the syntactic layer's spaCy parses are capped and batched through ``nlp.pipe``
  rather than one pipeline call per distinct name,
* nickname matching consults a prefix bucket instead of scanning every person
  for every person.

spaCy is stubbed throughout: the reduction itself is covered by
``test_syntactic_merge.py``, and what is measured here is call counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.consolidation import entity_resolution as er


class CountingStore:
    """In-memory store that records how often each primitive is called."""

    def __init__(self, entities: list[dict[str, Any]], speakers: list[str]) -> None:
        self.entities: dict[str, dict[str, Any]] = {e["name_norm"]: dict(e) for e in entities}
        self.speakers = list(speakers)
        self.merges: list[tuple[str, str]] = []
        self.calls: dict[str, int] = {}

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    async def list_entities(self) -> list[dict[str, Any]]:
        self._count("list_entities")
        return [dict(e) for e in self.entities.values()]

    async def distinct_speakers(self) -> list[str]:
        self._count("distinct_speakers")
        return list(self.speakers)

    async def drop_entities(self, name_norms: list[str]) -> int:
        self._count("drop_entities")
        return sum(self.entities.pop(n, None) is not None for n in name_norms)

    async def merge_entities(self, winner_norm: str, loser_norm: str) -> bool:
        self._count("merge_entities")
        if winner_norm == loser_norm or winner_norm not in self.entities:
            return False
        loser = self.entities.pop(loser_norm, None)
        if loser is None:
            return False
        win = self.entities[winner_norm]
        win["mention_count"] = win.get("mention_count", 0) + loser.get("mention_count", 0)
        self.merges.append((winner_norm, loser_norm))
        return True


class StubNLP:
    """spaCy stand-in that records the surfaces handed to ``pipe``."""

    def __init__(self) -> None:
        self.parsed: list[str] = []

    def pipe(self, texts: Any) -> Any:
        for text in texts:
            self.parsed.append(text)
            yield text  # a doc stand-in; _reduce_to_head is stubbed alongside


@pytest.fixture
def nlp(monkeypatch: pytest.MonkeyPatch) -> StubNLP:
    """Install the stub parser and neutralise the reduction (no grouping)."""
    stub = StubNLP()
    monkeypatch.setattr(er, "_parser", lambda: stub)
    monkeypatch.setattr(er, "_reduce_to_head", lambda doc: "")
    return stub


def _entity(name: str, **kw: Any) -> dict[str, Any]:
    return {"name_norm": name.lower(), "name": name, "type": "", "mention_count": 1, **kw}


@pytest.mark.asyncio
async def test_entity_table_and_speakers_are_read_once_per_flush(nlp: StubNLP) -> None:
    store = CountingStore(
        entities=[
            _entity("Melanie", type="PERSON", mention_count=40),
            _entity("Mel", type="PERSON", mention_count=6),
            _entity("others", mention_count=9),
            _entity("paths", mention_count=2),
            _entity("path", mention_count=5),
        ],
        speakers=["melanie"],
    )

    await er.resolve_entities(store, "conv-26")

    assert store.calls["list_entities"] == 1, "one flush must read the entity table once"
    assert store.calls["distinct_speakers"] == 1


@pytest.mark.asyncio
async def test_the_single_fetch_still_shrinks_between_layers(nlp: StubNLP) -> None:
    """Later layers must see the earlier layers' drops and merges, not the raw table."""
    store = CountingStore(
        entities=[
            # Layer 1 drops "one"; were the set stale, layer 4 would still see it
            # as a 3-char alphabetic person-ish prefix of "ones".
            _entity("one", type="PERSON", mention_count=3),
            _entity("Melanie", type="PERSON", mention_count=40),
            _entity("Mel", type="PERSON", mention_count=6),
            _entity("paths", mention_count=2),
            _entity("path", mention_count=5),
        ],
        speakers=["melanie"],
    )

    result = await er.resolve_entities(store, "conv-26")

    assert result["considered"] == 5
    assert result["dropped"] == 1 and "one" not in store.entities
    # Layer 2 collapsed the plural, layer 4 the nickname — both off the one fetch.
    assert ("path", "paths") in store.merges
    assert ("melanie", "mel") in store.merges
    assert set(store.entities) == {"melanie", "path"}
    assert result["merged"] == 2


@pytest.mark.asyncio
async def test_syntactic_parse_is_capped_and_batched(
    nlp: StubNLP, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-flush spaCy cost is bounded, and spent on the busiest names."""
    monkeypatch.setattr(er, "_MAX_HEAD_PARSE", 3)
    store = CountingStore(
        entities=[_entity(f"thing{i}", mention_count=i) for i in range(20)],
        speakers=[],
    )

    await er.resolve_entities(store, "s1")

    assert len(nlp.parsed) == 3, "the parse must be capped, not one per distinct name"
    assert set(nlp.parsed) == {"thing19", "thing18", "thing17"}


class _CountingStr(str):
    """A name_norm that counts every prefix comparison made against it."""

    comparisons = 0

    def startswith(self, prefix: Any, *args: Any) -> bool:  # type: ignore[override]
        _CountingStr.comparisons += 1
        return str.startswith(self, prefix, *args)


def _person(norm: str, mentions: int) -> dict[str, Any]:
    return {
        "name_norm": _CountingStr(norm),
        "name": norm.capitalize(),
        "type": "PERSON",
        "mention_count": mentions,
    }


@pytest.mark.asyncio
async def test_nickname_matching_does_not_compare_every_person_pair(nlp: StubNLP) -> None:
    """O(P^2) prefix scanning is replaced by a lookup into a first-3-chars bucket."""
    # Distinct alphabetic names, no two sharing their first three characters.
    crowd = [_person(f"{chr(97 + i // 26)}{chr(97 + i % 26)}xnameholder", 2) for i in range(120)]
    store = CountingStore(
        entities=[*crowd, _person("melanie", 40), _person("mel", 6)],
        speakers=[],
    )
    _CountingStr.comparisons = 0

    result = await er.resolve_entities(store, "conv-26")

    # The real nickname is still found — the bucket narrows the search, not the answer.
    assert ("melanie", "mel") in store.merges
    assert result["merged"] == 1
    # Scanning every person for every person would be ~14_000 comparisons here.
    assert _CountingStr.comparisons < len(crowd), (
        f"{_CountingStr.comparisons} prefix comparisons for {len(crowd) + 2} persons "
        "— the bucket is not being used"
    )
