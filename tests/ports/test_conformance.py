"""Port conformance: the bundled ArcadeDB stores satisfy the public protocols.

runtime_checkable Protocols verify method *presence*, so these isinstance
checks fail loudly if a store drops a method the port contract advertises (or
a port lists a method no backend implements).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore
from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.storage.base import IGraphStore, ILTMStore, ISTMStore


def _client() -> MagicMock:
    return MagicMock()


def test_graph_store_satisfies_igraphstore() -> None:
    """The live contract: one graph store per namespace serves the whole system."""
    assert isinstance(GraphStore(client=_client(), db="db"), IGraphStore)


def test_arcadedb_stm_store_satisfies_istmstore() -> None:
    assert isinstance(ArcadeDBSTMStore(client=_client()), ISTMStore)


def test_arcadedb_ltm_store_satisfies_iltmstore() -> None:
    assert isinstance(ArcadeDBLTMStore(client=_client()), ILTMStore)
