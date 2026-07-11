"""Benchmark dispatcher: ``python -m evaluation <benchmark> [flags]``.

Benchmarks: ``locomo`` (default), ``longmemeval``, ``beam``.
Bare flags (``python -m evaluation --limit 5``) run locomo for backward
compatibility with existing tooling.
"""

from __future__ import annotations

import sys

_BENCHMARKS = ("locomo", "longmemeval", "beam")


def main() -> None:
    argv = sys.argv[1:]
    benchmark = "locomo"
    if argv and argv[0] in _BENCHMARKS:
        benchmark, argv = argv[0], argv[1:]
    elif argv and not argv[0].startswith("-"):
        raise SystemExit(
            f"Unknown benchmark {argv[0]!r}. Choose from: {', '.join(_BENCHMARKS)}"
        )

    if benchmark == "locomo":
        from evaluation.locomo.cli import main as benchmark_main
    elif benchmark == "longmemeval":
        from evaluation.longmemeval.cli import main as benchmark_main
    else:
        from evaluation.beam.cli import main as benchmark_main
    benchmark_main(argv)


if __name__ == "__main__":
    main()
