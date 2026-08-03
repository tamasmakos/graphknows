"""CLI: ``graphknows-topics`` — rebuild or sweep the topic layer.

Rebuild one namespace's topics at chosen settings, or sweep a granularity range
and print the metric table so the setting is picked from evidence rather than
taste.

Also runnable as ``python -m graphknows.cli.topics``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from graphknows.settings import GraphKnowsSettings, TopicMode
from graphknows.topics.config import TopicConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the topic CLI argument parser."""
    p = argparse.ArgumentParser(prog="graphknows-topics", description=__doc__)
    p.add_argument("--namespace", default="", help="Tenant namespace (default: configured).")
    p.add_argument(
        "--mode",
        choices=[m.value for m in TopicMode],
        default="extractive",
        help="extractive (no LLM) | llm | none.",
    )
    p.add_argument("--min-size", type=int, default=2, help="Drop clusters below this size.")
    p.add_argument("--export", default="", help="Write a read-only JSON artifact here.")
    p.add_argument("--no-persist", action="store_true", help="Compute and report, do not write.")
    p.add_argument(
        "--sweep",
        default="",
        help=(
            "Comma-separated --min-size values to compare, e.g. 2,3,5,10. Never "
            "persists. min_size is the only knob: the grouping is the Louvain "
            "partition, whose parameters ArcadeDB ignores."
        ),
    )
    return p


_COLUMNS = (
    ("topics", "topics"),
    ("mean_size", "mean size"),
    ("largest_topic", "largest"),
    ("newman_modularity", "newman Q"),
    ("mean_coherence", "coherence"),
    ("chunk_coverage", "chunk cov"),
    ("entity_coverage", "ent cov"),
)


async def amain(argv: list[str] | None = None) -> int:
    """Rebuild or sweep the topic layer. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    from graphknows.memory import Memory

    settings = GraphKnowsSettings()
    base = TopicConfig(min_size=args.min_size, mode=TopicMode(args.mode))

    mem = Memory(settings, namespace=args.namespace or None)
    try:
        if args.sweep:
            raw = [x for x in args.sweep.split(",") if x.strip()]
            print(f"{'min_size':>10}  " + "  ".join(f"{h:>10}" for _, h in _COLUMNS))
            print("-" * (12 + 12 * len(_COLUMNS)))
            for text in raw:
                value = int(text)
                cfg = replace(base, min_size=value)
                out = await mem.build_topics(config=cfg, persist=False)
                print(f"{value:>10}  " + "  ".join(f"{out.get(k, 0)!s:>10}" for k, _ in _COLUMNS))
            print(
                "\nPick on newman Q + coherence, and check entity_coverage. The "
                "reported ArcadeDB modularity is NOT Newman's Q and rewards coarser "
                "partitions — do not tune on it."
            )
            return 0

        out = await mem.build_topics(
            config=base, persist=not args.no_persist, export_path=args.export
        )
        for key in sorted(out):
            print(f"  {key:38} {out[key]}")
        return 0
    finally:
        await mem.close()


def main() -> None:
    """Console entry point."""
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
