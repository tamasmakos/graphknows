"""There is no WordNet schema plane at all (issue #142).

Today ``graphknows/symbolic/ontology/senses.py`` reads the WordNet corpus at
MATCH time to pick a sense for lemma expansion (issue #141) and then throws
the synset away: nothing is ever written. ``_schema.py`` declares no
SYNSET/SENSE vertex and no HAS_SENSE edge, and ``GraphStore`` has no writer
for one. This pins the contract the rest of the plan has to satisfy:

- ``graphknows/symbolic/wordnet.py`` — pure domain logic (MFS baseline WSD,
  shortest hypernym ancestry, sense-level relations) with no I/O beyond the
  nltk corpus reader.
- ``GraphStore.write_senses`` / ``write_synset_hypernyms`` /
  ``write_sense_relations`` / ``synsets_missing_parents`` — the write choke
  points a lazy (grounding-driven) materialisation of WordNet's ~117k
  synsets routes through, instead of an eager bulk install.

Every test here fails today because ``graphknows/symbolic/wordnet.py`` does
not exist and ``GraphStore`` has none of the four methods above.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.symbolic.wordnet import (
    ID_PREFIX,
    MAX_HYPERNYM_DEPTH,
    NOUN_ONLY_RELATIONS,
    SENSE_RELATIONS,
    SYNSET_RELATIONS,
    TRANSITIVE_RELATIONS,
    ancestors,
    most_frequent_sense,
    synset_id,
)

pytestmark = pytest.mark.unit


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return client


def _command_strings(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


def _sense_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "Ada",
        "synset_id": "wn30:dog.n.01",
        "synset_name": "dog.n.01",
        "pos": "n",
        "definition": "a member of the genus Canis",
        "sense_id": "wn30:dog%1:05:00::",
        "sense_key": "dog%1:05:00::",
        "lemma": "dog",
        "chunk_id": "c1",
        "span": [0, 3],
        "channel": "wordnet",
        "version": "3.0",
        "matcher": "mfs",
        "score": 1.0,
        "rank": 0,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# 1. MFS baseline WSD + the lazy-mint bound (issue #142)
# ---------------------------------------------------------------------------


def test_most_frequent_sense_returns_first_noun_synset_with_bridged_sense_key() -> None:
    result = most_frequent_sense("dog")
    assert result is not None
    assert result["synset_name"] == "dog.n.01"
    assert str(result["synset_id"]).startswith(ID_PREFIX)
    sense_id = str(result["sense_id"])
    assert sense_id.startswith(ID_PREFIX)
    sense_key = sense_id[len(ID_PREFIX) :]
    # The sense KEY (not an nltk-internal ordinal) is the cross-release
    # bridge — WordNet sense keys are stable across corpus versions.
    assert "%" in sense_key


def test_most_frequent_sense_is_none_for_a_word_with_no_synsets() -> None:
    assert most_frequent_sense("asdkjhqwe") is None


# ---------------------------------------------------------------------------
# 2. Shortest ancestor chain, bounded (the lazy-mint contract)
# ---------------------------------------------------------------------------


def test_ancestors_is_the_shortest_hypernym_path_and_is_bounded() -> None:
    # Eager install would mint every synset up front (~117k for WordNet 3.0
    # noun+verb+adj+adv synsets); ancestors() must mint only the handful on
    # ONE shortest path from a single synset to a root.
    rows, truncated = ancestors("dog.n.01")

    assert truncated is False
    assert len(rows) <= MAX_HYPERNYM_DEPTH
    assert len(rows) < 1000, "ancestors() must mint a handful of synsets, not the whole corpus"

    from nltk.corpus import wordnet as wn  # type: ignore[import-untyped]

    paths = wn.synset("dog.n.01").hypernym_paths()
    shortest_len = min(len(p) for p in paths)
    # hypernym_paths() includes the synset itself; the edge chain from the
    # synset to its root has one fewer hop than the path's node count.
    assert len(rows) == shortest_len - 1

    # The chain's `child`/`parent` ids must be the SAME offset-based scheme
    # write_senses() uses for the SYNSET vertex a grounding mints
    # (synset_id()), never the synset's NAME -- otherwise write_synset_hypernyms'
    # MATCH on `child` cannot find the grounded synset and the chain never
    # attaches to anything (issue #142).
    expected_child = synset_id(wn.synset("dog.n.01"))
    for row in rows:
        assert row["child"] == expected_child
        assert str(row["parent"]).startswith(ID_PREFIX)
        expected_child = row["parent"]
    assert rows[-1]["parent_name"] == "entity.n.01"
    assert rows[-1]["parent"] == synset_id(wn.synset("entity.n.01"))


# ---------------------------------------------------------------------------
# 3. instance_hypernym must never be closed over transitively
# ---------------------------------------------------------------------------


def test_instance_hypernym_edge_is_not_transitive() -> None:
    rows, _truncated = ancestors("mount_everest.n.01")
    assert rows, "expected at least the instance_hypernym edge for a named instance"

    assert rows[0]["rel"] == "instance_hypernym"
    assert rows[0]["transitive"] is False
    for row in rows[1:]:
        assert row["rel"] == "hypernym"
        assert row["transitive"] is True

    assert "instance_hypernym" in SYNSET_RELATIONS
    assert "instance_hypernym" not in TRANSITIVE_RELATIONS


# ---------------------------------------------------------------------------
# 4. The synset/sense relation split is total and disjoint
# ---------------------------------------------------------------------------


def test_sense_and_synset_relation_sets_are_disjoint_and_exact() -> None:
    assert SENSE_RELATIONS.isdisjoint(SYNSET_RELATIONS)
    assert (
        frozenset({"antonym", "derivationally_related_form", "pertainym", "participle"})
        == SENSE_RELATIONS
    )
    meronyms = {"member_meronym", "part_meronym", "substance_meronym"}
    assert meronyms <= NOUN_ONLY_RELATIONS
    assert meronyms <= SYNSET_RELATIONS


# ---------------------------------------------------------------------------
# 5. Sense-level relations attach ONLY to SENSE nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_sense_relations_matches_only_sense_vertices() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")

    await store.write_sense_relations(
        [{"src": "wn30:a%1:00:00::", "dst": "wn30:b%1:00:00::", "rel": "antonym"}]
    )

    client.command.assert_awaited_once()
    stmt = client.command.await_args.args[1]
    assert ":SENSE" in stmt
    # Structural enforcement: the MATCH pattern binds BOTH endpoints as SENSE
    # vertices, so this writer cannot be pointed at a SYNSET by mistake.
    assert stmt.count(":SENSE") >= 2
    assert "SYNSET" not in stmt


# ---------------------------------------------------------------------------
# 6. HAS_SENSE uniqueness is (entity, chunk_id, span, synset)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_senses_keys_has_sense_on_chunk_span_and_synset() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")

    rows = [
        _sense_row(chunk_id="c1"),
        _sense_row(chunk_id="c2"),
    ]
    await store.write_senses(rows)

    payload_rows = [
        r for call in client.command.await_args_list for r in call.kwargs["params"]["rows"]
    ]
    chunk_ids = {r["chunk_id"] for r in payload_rows}
    assert chunk_ids == {"c1", "c2"}, "both distinct-chunk rows must reach the UNWIND payload"

    has_sense_stmts = [s for s in _command_strings(client) if "HAS_SENSE" in s]
    assert has_sense_stmts, "expected a HAS_SENSE edge to be written"
    stmt = has_sense_stmts[0]
    merge_idx = stmt.index("MERGE")
    has_sense_idx = stmt.index("HAS_SENSE", merge_idx)
    open_brace = stmt.index("{", has_sense_idx)
    close_brace = stmt.index("}", open_brace)
    pattern_body = stmt[open_brace:close_brace]
    # chunk_id, span and synset must be inside the HAS_SENSE {...} pattern
    # braces so they participate in the MERGE key, not only set afterward.
    assert "chunk_id" in pattern_body
    assert "span" in pattern_body
    assert "synset" in pattern_body


# ---------------------------------------------------------------------------
# 7. Provenance is asserted at the choke point, same contract as write_groundings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_senses_row_missing_chunk_id_raises() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    row = _sense_row()
    del row["chunk_id"]

    with pytest.raises(ValueError, match="chunk_id"):
        await store.write_senses([row])
