"""Port conformance: the bundled ArcadeDB stores satisfy the public protocols.

runtime_checkable Protocols verify method *presence*, so these isinstance
checks fail loudly if a store drops a method the port contract advertises (or
a port lists a method no backend implements).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore
from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.storage.arcadedb.vector_store import ArcadeDBVectorStore
from graphknows.storage.base import ILTMStore, ISTMStore, IVectorStore


def _client() -> MagicMock:
    return MagicMock()


def test_arcadedb_stm_store_satisfies_istmstore() -> None:
    assert isinstance(ArcadeDBSTMStore(client=_client()), ISTMStore)


def test_arcadedb_ltm_store_satisfies_iltmstore() -> None:
    assert isinstance(ArcadeDBLTMStore(client=_client()), ILTMStore)


def test_arcadedb_vector_store_satisfies_ivectorstore() -> None:
    assert isinstance(ArcadeDBVectorStore(client=_client()), IVectorStore)
