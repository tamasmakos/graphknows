"""Triple-level consolidation over REL edges (issue #140, B7 + B10).

``GraphStore.write_relation``/``write_frame_instances`` MERGE on the specific
relex predicate label, so the same fact arriving under two different verbs
(``hasSpouse`` vs. ``isMarriedTo``) never collapses, and a fact that
contradicts a functional-property edge (a second spouse) never invalidates
the first. ``consolidate_relations()`` is the namespace-wide pass that groups
current REL edges on ``(head, canonical_predicate, tail, polarity, modality,
valid_at)`` and rewrites them: duplicates merge with a raised ``support``
counter, and contradictions (differing polarity, or a second tail on a
functional predicate) tombstone the loser and record a ``CONTRADICTS`` link.

Mocking style follows tests/storage/test_graph_store.py: a MagicMock client
with async ``command``/``query``, and assertions read the emitted command
strings/params rather than internal call sequencing.

REL row keys below (``h``, ``h_name``, ``t``, ``t_name``, ``cp``, ``p``,
``pol``, ``mod``, ``va``, ``cur``, ``conf``, ``ev``, ``support``,
``created_at``) are the contract the implementer's REL-fetch query must
return — keep them stable if you change the query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore

_PREDICATES = ("HAS_SPOUSE", "IS_MARRIED_TO")


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return client


def _command_strings(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


def _commands(client: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    """(statement, params) pairs for every ``command`` call, in call order."""
    return [
        (call.args[1], call.kwargs.get("params") or {}) for call in client.command.await_args_list
    ]


def _query_stub(
    ontology_rows: list[dict[str, Any]], rel_rows: list[dict[str, Any]]
) -> Callable[..., Any]:
    """Order-insensitive ``client.query`` stub: routed by statement content."""

    def _side_effect(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        stmt = args[1] if len(args) > 1 else kwargs.get("command", "")
        if "ONTOLOGY_PROPERTY" in stmt.upper():
            return ontology_rows
        return rel_rows

    return _side_effect


def _touches(stmt: str, params: dict[str, Any], predicate: str) -> bool:
    """Whether a command references ``predicate``, in the SQL or its params."""
    return predicate in stmt or predicate in repr(params)


def _mentions_tail(stmt: str, params: dict[str, Any], tail: str) -> bool:
    return tail in stmt or tail in repr(params)


# ---------------------------------------------------------------------------
# 1. Dedup: same fact, two predicate surfaces sharing one canonical_predicate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_merges_duplicate_surfaces_and_raises_support() -> None:
    client = _mock_client()
    rel_rows = [
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "spouse",
            "p": "HAS_SPOUSE",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c1"],
            "support": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "spouse",
            "p": "IS_MARRIED_TO",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.85,
            "ev": ["c2"],
            "support": 1,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    ]
    client.query = AsyncMock(side_effect=_query_stub(ontology_rows=[], rel_rows=rel_rows))

    await GraphStore(client=client).consolidate_relations()

    commands = _commands(client)
    delete_cmds = [(s, p) for s, p in commands if "DELETE" in s.upper()]
    assert len(delete_cmds) == 1, "exactly one duplicate edge should be deleted"

    loser = next(pred for pred in _PREDICATES if _touches(*delete_cmds[0], pred))
    survivor = next(pred for pred in _PREDICATES if pred != loser)

    survivor_cmds = [
        (s, p) for s, p in commands if "SUPPORT" in s.upper() and _touches(s, p, survivor)
    ]
    assert survivor_cmds, "the surviving edge should have its support counter raised"
    assert any(2 in p.values() for _, p in survivor_cmds), "support should be raised to 2"

    # Nothing else is deleted.
    assert len(delete_cmds) == 1


# ---------------------------------------------------------------------------
# 2. Contradiction by polarity — no delete, both edges kept, linked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polarity_contradiction_links_without_deleting() -> None:
    client = _mock_client()
    rel_rows = [
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "spouse",
            "p": "HAS_SPOUSE",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c1"],
            "support": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "spouse",
            "p": "HAS_SPOUSE",
            "pol": "neg",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.8,
            "ev": ["c2"],
            "support": 1,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    ]
    client.query = AsyncMock(side_effect=_query_stub(ontology_rows=[], rel_rows=rel_rows))

    await GraphStore(client=client).consolidate_relations()

    commands = _commands(client)
    assert not [s for s, _ in commands if "DELETE" in s.upper()], (
        "a polarity contradiction must not delete either edge"
    )
    contradicts_cmds = [
        s for s, _ in commands if "MERGE" in s.upper() and "CONTRADICTS" in s.upper()
    ]
    assert contradicts_cmds, "a CONTRADICTS link should be written between the two edges"


# ---------------------------------------------------------------------------
# 3. Functional collision (ontology-declared) — older tail tombstoned.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_functional_property_collision_tombstones_older_edge() -> None:
    client = _mock_client()
    rel_rows = [
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "spouse",
            "p": "HAS_SPOUSE",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c1"],
            "support": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "beth",
            "t_name": "Beth",
            "cp": "spouse",
            "p": "HAS_SPOUSE",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c2"],
            "support": 1,
            "created_at": "2026-02-01T00:00:00+00:00",
        },
    ]
    ontology_rows = [{"label": "spouse", "uri": "https://schema.org/spouse"}]
    client.query = AsyncMock(
        side_effect=_query_stub(ontology_rows=ontology_rows, rel_rows=rel_rows)
    )

    await GraphStore(client=client).consolidate_relations()

    commands = _commands(client)
    assert not [s for s, _ in commands if "DELETE" in s.upper()], (
        "the older edge is tombstoned, not deleted"
    )

    tombstone_cmds = [
        (s, p) for s, p in commands if "IS_CURRENT" in s.upper() and "INVALID_AT" in s.upper()
    ]
    older_tombstoned = [c for c in tombstone_cmds if _mentions_tail(*c, "alice")]
    newer_tombstoned = [c for c in tombstone_cmds if _mentions_tail(*c, "beth")]
    assert older_tombstoned, "the older (alice) edge should be marked invalid"
    assert not newer_tombstoned, "the newer (beth) edge should stay live"

    contradicts_cmds = [
        s for s, _ in commands if "MERGE" in s.upper() and "CONTRADICTS" in s.upper()
    ]
    assert contradicts_cmds, "the two spouse edges should be linked as contradicting"


# ---------------------------------------------------------------------------
# 4. Functional collision (hand-list fallback, no ontology row).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_valued_fallback_tombstones_older_edge_without_ontology() -> None:
    client = _mock_client()
    rel_rows = [
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "alice",
            "t_name": "Alice",
            "cp": "lemma:married_to",
            "p": "MARRIED_TO",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c1"],
            "support": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "h": "jon",
            "h_name": "Jon",
            "t": "beth",
            "t_name": "Beth",
            "cp": "lemma:married_to",
            "p": "MARRIED_TO",
            "pol": "pos",
            "mod": "none",
            "va": None,
            "cur": True,
            "conf": 0.9,
            "ev": ["c2"],
            "support": 1,
            "created_at": "2026-02-01T00:00:00+00:00",
        },
    ]
    # No ontology row declares `lemma:married_to` functional — the hand-coded
    # _SINGLE_VALUED_RELATIONS frozenset (MARRIED_TO) must drive this instead.
    client.query = AsyncMock(side_effect=_query_stub(ontology_rows=[], rel_rows=rel_rows))

    await GraphStore(client=client).consolidate_relations()

    commands = _commands(client)
    assert not [s for s, _ in commands if "DELETE" in s.upper()], (
        "the older edge is tombstoned, not deleted"
    )

    tombstone_cmds = [
        (s, p) for s, p in commands if "IS_CURRENT" in s.upper() and "INVALID_AT" in s.upper()
    ]
    older_tombstoned = [c for c in tombstone_cmds if _mentions_tail(*c, "alice")]
    newer_tombstoned = [c for c in tombstone_cmds if _mentions_tail(*c, "beth")]
    assert older_tombstoned, "the older (alice) edge should be marked invalid"
    assert not newer_tombstoned, "the newer (beth) edge should stay live"

    contradicts_cmds = [
        s for s, _ in commands if "MERGE" in s.upper() and "CONTRADICTS" in s.upper()
    ]
    assert contradicts_cmds, "the two married_to edges should be linked as contradicting"
