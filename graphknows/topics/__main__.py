"""CLI: ``python -m graphknows.topics`` — rebuild or sweep the topic layer.

Rebuild one namespace's topics at chosen settings, or sweep a granularity range
and print the metric table so the setting is picked from evidence rather than
taste.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from graphknows.settings import GraphKnowsSettings, TopicMode
from graphknows.topics.config import TopicConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the topic CLI argument parser."""
    p = argparse.ArgumentParser(prog="python -m graphknows.topics", description=__doc__)
    p.add_argument("--namespace", default="", help="Tenant namespace (default: configured).")
    p.add_argument(
        "--mode",
        choices=[m.value for m in TopicMode],
        default="extractive",
        help="extractive (no LLM) | llm | none.",
    )
    p.add_argument(
        "--source",
        choices=["community", "graph_embedding"],
        default="community",
        help=(
            "What to cluster. 'community' covers ~every entity but has no "
            "granularity knob; 'graph_embedding' is tunable but only covers "
            "entities with >=3 relation edges — check ent cov before trusting it."
        ),
    )
    p.add_argument("--distance", type=float, default=0.35, help="Granularity knob (0-1).")
    p.add_argument("--min-size", type=int, default=2, help="Drop clusters below this size.")
    p.add_argument("--n2v-walk-length", type=int, default=20)
    p.add_argument("--n2v-walks", type=int, default=20)
    p.add_argument("--n2v-dimensions", type=int, default=32)
    p.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Re-run node2vec with the parameters above before clustering.",
    )
    p.add_argument("--export", default="", help="Write a read-only JSON artifact here.")
    p.add_argument("--no-persist", action="store_true", help="Compute and report, do not write.")
    p.add_argument(
        "--sweep",
        default="",
        help="Comma-separated values to compare, e.g. 0.2,0.3,0.4. Never persists.",
    )
    p.add_argument(
        "--sweep-param",
        choices=["distance", "min_size"],
        default="distance",
        help=(
            "Which knob --sweep varies. NOTE: 'distance' only has an effect with "
            "--source graph_embedding; the community source is partitioned by "
            "Louvain, so sweep min_size there."
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
    base = TopicConfig(
        source=args.source,
        distance=args.distance,
        min_size=args.min_size,
        mode=TopicMode(args.mode),
        n2v_walk_length=args.n2v_walk_length,
        n2v_walks=args.n2v_walks,
        n2v_dimensions=args.n2v_dimensions,
        recompute_embeddings=args.recompute_embeddings,
    )

    mem = Memory(settings, namespace=args.namespace or None)
    try:
        if args.sweep:
            param = args.sweep_param
            if param == "distance" and args.source == "community":
                print(
                    "warning: --source community is partitioned by Louvain, whose "
                    "parameters ArcadeDB ignores, so 'distance' changes nothing. "
                    "Sweep --sweep-param min_size instead.\n"
                )
            raw = [x for x in args.sweep.split(",") if x.strip()]
            print(f"{param:>10}  " + "  ".join(f"{h:>10}" for _, h in _COLUMNS))
            print("-" * (12 + 12 * len(_COLUMNS)))
            for text in raw:
                # Explicit per-knob replace(): a **{param: value} splat is
                # untypable, and there are only two knobs.
                value: float | int
                if param == "distance":
                    value = float(text)
                    cfg = replace(base, distance=value)
                else:
                    value = int(text)
                    cfg = replace(base, min_size=value)
                out = await mem.build_topics(config=cfg, persist=False)
                print(
                    f"{value:>10}  "
                    + "  ".join(f"{out.get(k, 0)!s:>10}" for k, _ in _COLUMNS)
                )
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
