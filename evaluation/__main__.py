"""Command dispatcher: ``python -m evaluation <command> [flags]``.

Benchmarks: ``locomo`` (default), ``longmem`` (LongMemEval), ``beam`` (BEAM) and
``repo`` (this repository's own agent transcripts).
Also ``audit`` — the stratified triplet audit, which measures graph quality
directly rather than through a benchmark — and the harness commands
``baseline``, ``panel``, ``deadweight``, ``identity``, ``scenario`` and
``download``.
Bare flags (``python -m evaluation --limit 5``) run locomo for backward
compatibility with existing tooling. Each benchmark ships a self-contained
``cli.py``; all share ``evaluation.common`` (ingest → retrieve → DSPy judge).
"""

from __future__ import annotations

import importlib
import sys

# command name -> the module whose `main(argv)` drives it. Only `locomo`,
# `longmem`, `beam` and `repo` are benchmarks; the rest measure the core directly
# (graph quality, panel gate, dead weight, identity, scenarios) or provision
# corpora. They all dispatch identically, so one table serves them all rather
# than growing a second entry point.
_COMMANDS = {
    "locomo": "evaluation.locomo.cli",
    "longmem": "evaluation.longmem.cli",
    "beam": "evaluation.beam.cli",
    "repo": "evaluation.repo.cli",
    "audit": "evaluation.audit.cli",
    "baseline": "evaluation.common.baseline",
    "panel": "evaluation.common.panel",
    "deadweight": "evaluation.deadweight",
    "identity": "evaluation.identity",
    "scenario": "evaluation.scenarios",
    "download": "evaluation.common.download",
}
_DEFAULT = "locomo"


def main() -> None:
    argv = sys.argv[1:]
    command = _DEFAULT
    if argv and argv[0] in _COMMANDS:
        command, argv = argv[0], argv[1:]
    elif argv and not argv[0].startswith("-"):
        raise SystemExit(f"Unknown command {argv[0]!r}. Choose from: {', '.join(_COMMANDS)}")

    command_main = importlib.import_module(_COMMANDS[command]).main
    command_main(argv)


if __name__ == "__main__":
    main()
