"""The production secret check and the env value that switches it on.

``check_production_secrets`` is a no-op for every env value but ``production``,
so a misspelling silently disables it. ``graphknows_env`` is a lowercase enum
(FR-006, SC-004); this pins that the near-misses are rejected at construction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphknows.exceptions import ConfigurationError
from graphknows.settings import Env, GraphKnowsSettings


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own shell must not steer these."""
    monkeypatch.delenv("GRAPHKNOWS_ENV", raising=False)


def test_env_value_is_a_lowercase_enum() -> None:
    """``prod``, ``Production`` and ``PROD`` raise instead of skipping the check."""
    for spelling in ("prod", "Production", "PROD", "Development", "staging"):
        with pytest.raises(ValidationError):
            GraphKnowsSettings(_env_file=None, GRAPHKNOWS_ENV=spelling)

    assert GraphKnowsSettings(_env_file=None).graphknows_env is Env.development
    assert (
        GraphKnowsSettings(
            _env_file=None,
            GRAPHKNOWS_ENV="production",
            GRAPHKNOWS_ARCADEDB_PASSWORD="not-the-default",
        ).graphknows_env
        is Env.production
    )


def test_default_password_under_production_raises() -> None:
    """The development password is refused once the env says production."""
    with pytest.raises(ConfigurationError, match="GRAPHKNOWS_ARCADEDB_PASSWORD"):
        GraphKnowsSettings(
            _env_file=None,
            GRAPHKNOWS_ENV="production",
            GRAPHKNOWS_ARCADEDB_PASSWORD="changeme",
        )
