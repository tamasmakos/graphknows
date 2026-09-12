"""`.importlinter` is the machine-check behind the declared architecture.

CI runs ``lint-imports``; this pins the contracts a rename or a merge could
silently drop, and asserts the whole set still holds.
"""

from __future__ import annotations

import configparser
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[1] / ".importlinter"


@pytest.fixture(scope="module")
def config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(CONFIG_PATH, encoding="utf-8")
    return parser


def _modules(config: configparser.ConfigParser, contract: str, key: str) -> list[str]:
    return config[f"importlinter:contract:{contract}"][key].split()


def test_packs_layer_sits_between_memory_and_the_pipelines(
    config: configparser.ConfigParser,
) -> None:
    layers = _modules(config, "layers", "layers")
    assert "graphknows.packs" in layers, "the packs layer is undeclared"
    assert layers.index("graphknows.memory") < layers.index("graphknows.packs")
    assert layers.index("graphknows.packs") < layers.index("graphknows.ingestion")


def test_core_knows_no_domain_forbids_packs_to_every_core_package(
    config: configparser.ConfigParser,
) -> None:
    sources = _modules(config, "core-knows-no-domain", "source_modules")
    assert sources == [
        "graphknows.ingestion",
        "graphknows.retrieval",
        "graphknows.channels",
        "graphknows.symbolic",
        "graphknows.storage",
    ]
    assert _modules(config, "core-knows-no-domain", "forbidden_modules") == ["graphknows.packs"]


def test_claude_code_hooks_are_held_to_the_stdlib_only_contract(
    config: configparser.ConfigParser,
) -> None:
    sources = _modules(config, "client-stdlib-only", "source_modules")
    assert "graphknows.integrations.claude_code" in sources


def test_lint_imports_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from importlinter.cli import lint_imports_command as c; c()"],
        cwd=CONFIG_PATH.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
