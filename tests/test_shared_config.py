from __future__ import annotations

import os
import sys

import pytest

# Ensure repo root and shared package are importable.
_REPO = os.path.join(os.path.dirname(__file__), "..")
_SHARED = os.path.join(_REPO, "libs", "python", "graphknows-shared")
for _p in (_REPO, _SHARED):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_base_service_settings_uses_repo_env_by_default():
    from graphknows.shared.config import BaseServiceSettings

    settings = BaseServiceSettings()

    assert settings.pgvector_dsn == "postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb"
    assert settings.neo4j_password == "testpassword123"


def test_base_service_settings_preserves_explicit_pgvector_dsn(monkeypatch: pytest.MonkeyPatch):
    from graphknows.shared.config import BaseServiceSettings

    monkeypatch.setenv("PGVECTOR_DSN", "postgresql://bob:dsn-pass@dsn-host:5432/customdb")
    monkeypatch.setenv("PGVECTOR_PASSWORD", "ignored-pass")

    settings = BaseServiceSettings()

    assert settings.pgvector_dsn == "postgresql://bob:dsn-pass@dsn-host:5432/customdb"
