"""Benchmark dispatcher: ``python -m evaluation <benchmark> [flags]``.

Benchmarks: ``locomo`` (default), ``longmem`` (LongMemEval), ``beam`` (BEAM).
Also ``audit`` — the stratified triplet audit, which measures graph quality
directly rather than through a benchmark.
Bare flags (``python -m evaluation --limit 5``) run locomo for backward
compatibility with existing tooling. Each benchmark ships a self-contained
``cli.py``; all share ``evaluation.common`` (ingest → retrieve → DSPy judge).
"""

from __future__ import annotations

import importlib
import sys

# benchmark name -> the module whose `main(argv)` drives it. `audit` is not a
# benchmark — it is the stratified triplet audit, which measures graph quality
# without routing through retrieval — but it dispatches identically, so it lives
# here rather than growing a second entry point.
_BENCHMARKS = {
    "locomo": "evaluation.locomo.cli",
    "longmem": "evaluation.longmem.cli",
    "beam": "evaluation.beam.cli",
    "audit": "evaluation.audit.cli",
}
_DEFAULT = "locomo"


def main() -> None:
    argv = sys.argv[1:]
    benchmark = _DEFAULT
    if argv and argv[0] in _BENCHMARKS:
        benchmark, argv = argv[0], argv[1:]
    elif argv and not argv[0].startswith("-"):
        raise SystemExit(f"Unknown benchmark {argv[0]!r}. Choose from: {', '.join(_BENCHMARKS)}")

    benchmark_main = importlib.import_module(_BENCHMARKS[benchmark]).main
    benchmark_main(argv)


if __name__ == "__main__":
    main()
