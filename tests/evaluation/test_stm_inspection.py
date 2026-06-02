"""Unit tests for STM inspection helpers."""

from __future__ import annotations

from evaluation.core.stm_inspection import derive_inspect_command


def test_derive_inspect_command_from_graphknows_mcp() -> None:
    command = 'docker compose exec -T mcp graphknows-mcp'
    assert derive_inspect_command(command) == 'docker compose exec -T mcp python -'


def test_derive_inspect_command_returns_none_for_unknown_shape() -> None:
    assert derive_inspect_command("graphknows-mcp") == "python -"
    assert derive_inspect_command("custom-runner serve") is None
