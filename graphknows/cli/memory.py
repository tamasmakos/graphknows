"""``graphknows ingest`` and ``graphknows recall`` — the facade on argv."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from graphknows.memory import Memory
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source


def _read(path: Path) -> tuple[Source, list[Segment]]:
    """One file as the source and segments the facade ingests.

    ponytail: the whole file becomes one prose segment — route this through the
    parser registry (``graphknows.ingestion.parsers``) once a prose parser is
    registered there, so evidence is quoted at chunk granularity.
    """
    data = path.read_bytes()
    source = Source(uri=path.as_uri(), content_hash=sha256(data).hexdigest(), mime="text/plain")
    segment = Segment(
        source_id=source.id,
        text=data.decode("utf-8"),
        kind=SegmentKind.prose,
        byte_range=(0, len(data)),
    )
    return source, [segment]


async def _ingest(args: argparse.Namespace) -> dict[str, Any]:
    """Write the named file into memory and return the ingest report."""
    source, segments = _read(Path(args.path))
    async with Memory(namespace=args.namespace) as memory:
        report = await memory.ingest(source, segments)
    return report.model_dump(mode="json")


async def _recall(args: argparse.Namespace) -> dict[str, Any]:
    """Recall the facts the query's symbols activate, with their evidence."""
    async with Memory(namespace=args.namespace) as memory:
        result = await memory.recall(args.query)
    return result.model_dump(mode="json")


def _parser() -> argparse.ArgumentParser:
    """The argv grammar: one subcommand per facade operation."""
    parser = argparse.ArgumentParser(prog="graphknows", description=__doc__)
    parser.add_argument("--namespace", default=None, help="Namespace to act in.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Write one file into memory.")
    ingest.add_argument("path", help="File to ingest.")
    ingest.set_defaults(run=_ingest)
    recall = commands.add_parser("recall", help="Recall facts a query activates.")
    recall.add_argument("query", help="What to recall.")
    recall.set_defaults(run=_recall)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one facade command and print its report as JSON."""
    args = _parser().parse_args(argv)
    print(json.dumps(asyncio.run(args.run(args)), indent=2))
    return 0
