"""Guard: the STM ingest path stays async-first (no sync thread-bridge).

Phase 2 removed the `_async_bridge._run` hop from STM ingest so store I/O runs
on the caller's event loop. This test fails if the bridge creeps back into the
ingest modules.
"""

from __future__ import annotations

import inspect

from graphknows.memory.stm import dialog, ingest, service
from graphknows.memory.stm._entity_writer import (
    write_entity_nodes_and_edges,
    write_relation_edges,
)


def test_ingest_modules_do_not_import_async_bridge() -> None:
    for module in (service, ingest, dialog):
        assert "_async_bridge" not in inspect.getsource(module), (
            f"{module.__name__} must not use the sync thread-bridge"
        )


def test_public_ingest_methods_are_coroutines() -> None:
    assert inspect.iscoroutinefunction(service.STMService.ingest)
    assert inspect.iscoroutinefunction(service.STMService.ingest_messages)
    assert inspect.iscoroutinefunction(service.STMService.record_turn)
    assert inspect.iscoroutinefunction(service.STMService.connect)
    assert inspect.iscoroutinefunction(dialog.DialogCapture.write_turn)


def test_entity_writer_helpers_are_async() -> None:
    assert inspect.iscoroutinefunction(write_entity_nodes_and_edges)
    assert inspect.iscoroutinefunction(write_relation_edges)
