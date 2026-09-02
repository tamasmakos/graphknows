"""The SCHEMA_STAMP singleton and the version digest that fills it."""

from __future__ import annotations

import pytest

from graphknows.storage.arcadedb import _schema


def test_schema_version_changes_with_ddl_text_and_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _schema.schema_version(768)

    assert baseline == _schema.schema_version(768), "digest must be deterministic"
    assert baseline != _schema.schema_version(1024), "dims must reach the digest"

    monkeypatch.setattr(
        _schema,
        "_CORE_DDL",
        [*_schema._CORE_DDL, ("sql", "CREATE VERTEX TYPE LATER IF NOT EXISTS")],
    )
    assert baseline != _schema.schema_version(768), "DDL text must reach the digest"
