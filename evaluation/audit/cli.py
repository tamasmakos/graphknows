"""Stratified triplet audit: ``python -m evaluation audit sample|report``.

The primary graph-quality metric. It never touches the retriever, because
``evidence_recall`` cannot see graph-quality work: a run that added a whole
symbolic plane moved it by 0.0000. Instead a human judges a stratified sample of
REL edges against their evidence text, on four binary axes:

    span               are head/tail the right surface strings?
    predicate          is the relation right?
    direction          are head/tail the right way round?
    polarity_modality  is the assertion status right (asserted/negated/hedged)?

Two steps, with a human in between::

    # 1. draw the sheet (deterministic given --seed)
    python -m evaluation audit sample --namespace eval_locomo_llm_free_conv_30 \\
        -n 150 --seed 0

    # 2. a human fills the four judgement columns with 1/0, then
    python -m evaluation audit report --sheet evaluation/results/audit/<sheet>.csv

    # 3. or compare against an earlier report
    python -m evaluation audit report --sheet evaluation/results/audit/<sheet>.csv \\
        --baseline evaluation/audit/baseline/conv-30-seed0-report.json

``sample`` writes ``<stem>.csv`` (the sheet) plus ``<stem>.json`` (the manifest:
seed, database, per-tier populations and quota). ``report`` writes
``<stem>-report.json`` and ``<stem>-report.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.audit.core import (
    JUDGEMENT_COLUMNS,
    SHEET_COLUMNS,
    aggregate,
    build_report,
    compare,
    render_markdown,
)
from evaluation.audit.core import stratified_sample as stratify
from graphknows.settings import GraphKnowsSettings
from graphknows.storage.arcadedb.client import ArcadeDBClient
from graphknows.storage.namespace import db_name

DEFAULT_OUTPUT_DIR = Path("evaluation/results/audit")
DEFAULT_SAMPLE_SIZE = 150
_EVIDENCE_SEPARATOR = "\n--- chunk break ---\n"
# Some edges land with `evidence: []` and no frame instance — no provenance at
# all. Say so in the sheet instead of handing the judge an empty cell to guess at.
_NO_EVIDENCE = "<no evidence recorded on this edge - cannot be judged>"
# A frame instance is re-evoked in every chunk that repeats it, up to 24 on this
# corpus. All of them are listed in `evidence_chunk_ids`, but only the first few
# are rendered: a judge needs one passage that settles the triplet, not a
# transcript.
# ponytail: fixed cap, primary chunk first. If judges report that the shown
# chunks miss the assertion, rank by trigger-span overlap instead.
_MAX_EVIDENCE_CHUNKS = 3

# Assertion status a human needs for the polarity+modality axis. It lives on the
# frame instance the edge projects, so a blank column on the edge is filled from
# there rather than left for the judge to guess at.
_INSTANCE_FALLBACK = ("frame", "polarity", "modality", "epistemic", "hedged")

_INSTANCE_QUERY = """
SELECT id, chunk_id, frame, polarity, modality, epistemic, hedged, attributed_to,
       out('FRAME_EVOKED_IN').chunk_id AS evoked_chunks
FROM FRAME_INSTANCE WHERE id IN :ids
"""

# Only edges the system would actually serve. Invalidated ones (`is_current`
# false) have already been retired by the temporal layer, so their precision
# measures a graph nobody reads.
_REL_QUERY = """
SELECT @rid AS rid, predicate, canonical_predicate, anchor, reified, frame,
       frame_instance_id, head_role, tail_role, polarity, modality, epistemic,
       hedged, confidence, verifier_score, evidence,
       outV().name AS head, inV().name AS tail
FROM REL
WHERE is_current IS NULL OR is_current = true
"""


async def _fetch_edges(client: ArcadeDBClient, database: str) -> list[dict[str, Any]]:
    """Read every currently-valid REL edge with its endpoint names."""
    return await client.query(database, _REL_QUERY, language="sql")


async def _lookup(
    client: ArcadeDBClient, database: str, query: str, ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Run an ``IN :ids`` lookup, or nothing at all when there is nothing to look up.

    ponytail: one round trip for the whole id list. Sample sizes are in the
    hundreds; batch it if an audit ever draws tens of thousands of edges.
    """
    if not ids:
        return []
    return await client.query(database, query, language="sql", params={"ids": list(ids)})


def _instance_chunks(record: Mapping[str, Any]) -> list[str]:
    """Chunk ids a frame instance is evidenced in, primary first, order stable.

    ``FRAME_EVOKED_IN`` is the real provenance link and returns every chunk that
    re-evoked the instance, in storage order. ``chunk_id`` is the instance's own
    extraction site, so it leads; the rest follow sorted, so a re-run of the
    sampler produces a byte-identical sheet.
    """
    primary = str(record.get("chunk_id") or "")
    evoked = {str(cid) for cid in (record.get("evoked_chunks") or []) if cid}
    return ([primary] if primary else []) + sorted(evoked - {primary})


async def attach_evidence(
    client: ArcadeDBClient, database: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return *rows* with their evidence text and frame-instance context filled in.

    Two routes to the evidence, because a REL edge stores it in one of two places:

    * a relex edge carries ``evidence``, a list of chunk ids, directly;
    * a reified edge is an *entity-adjacency projection* of a frame instance and
      carries none - its provenance is one hop away, on the instance, reachable
      as ``FRAME_INSTANCE -[FRAME_EVOKED_IN]-> CHUNK``.

    Missing that second hop reads as "2579 of 2735 edges have no evidence" when
    almost all of them do. The instance also carries the assertion status and
    ``attributed_to``, which is what a human needs to judge the polarity+modality
    axis at all, so blank edge columns are backfilled from it.
    """
    frame_ids = sorted(
        {
            str(row["frame_instance_id"])
            for row in rows
            if not row.get("evidence") and row.get("frame_instance_id")
        }
    )
    instances = {
        str(record["id"]): record
        for record in await _lookup(client, database, _INSTANCE_QUERY, frame_ids)
    }

    per_row: list[list[str]] = []
    for row in rows:
        chunk_ids = [str(cid) for cid in (row.get("evidence") or [])]
        if not chunk_ids and (instance := instances.get(str(row.get("frame_instance_id") or ""))):
            chunk_ids = _instance_chunks(instance)
        per_row.append(chunk_ids)

    shown = sorted({cid for chunk_ids in per_row for cid in chunk_ids[:_MAX_EVIDENCE_CHUNKS]})
    chunk_text = {
        str(record["chunk_id"]): str(record.get("text") or "")
        for record in await _lookup(
            client,
            database,
            "SELECT chunk_id, text FROM CHUNK WHERE chunk_id IN :ids",
            shown,
        )
    }

    enriched: list[dict[str, Any]] = []
    for row, chunk_ids in zip(rows, per_row, strict=True):
        record = dict(row)
        instance = instances.get(str(row.get("frame_instance_id") or "")) or {}
        for column in _INSTANCE_FALLBACK:
            if record.get(column) in (None, "") and instance.get(column) is not None:
                record[column] = instance[column]
        record["attributed_to"] = instance.get("attributed_to", "")

        record["evidence_chunk_ids"] = " ".join(chunk_ids)
        passages = [
            chunk_text.get(cid, f"<chunk {cid} not found>")
            for cid in chunk_ids[:_MAX_EVIDENCE_CHUNKS]
        ]
        if (elided := len(chunk_ids) - len(passages)) > 0:
            passages.append(f"<{elided} further chunk(s) evoke this frame instance; ids listed>")
        record["evidence_text"] = _EVIDENCE_SEPARATOR.join(passages) or _NO_EVIDENCE
        enriched.append(record)
    return enriched


def _write_sheet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the human-judgeable CSV: one edge per row, judgement columns blank."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})


def _read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_manifest(sheet: Path, explicit: Path | None) -> dict[str, Any] | None:
    """Load the sample manifest that pairs with *sheet*, if it is still around."""
    path = explicit or sheet.with_suffix(".json")
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"manifest {path} is not a JSON object")
    return loaded


async def _sample(args: argparse.Namespace) -> int:
    settings = GraphKnowsSettings()
    database = args.database or db_name(args.namespace)
    client = ArcadeDBClient(
        settings.arcadedb_url,
        settings.arcadedb_user,
        settings.arcadedb_password.get_secret_value(),
    )
    await client.connect()
    try:
        edges = await _fetch_edges(client, database)
        if not edges:
            raise SystemExit(f"no current REL edges in {database} — nothing to audit")
        # Over the full population, not the drawn sample: this is the vocabulary-usage
        # number an A/B compares, not a judging concern.
        predicate_counts = Counter(edge.get("canonical_predicate") or "<none>" for edge in edges)
        picked, census, quota = stratify(edges, args.size, args.seed)
        rows = await attach_evidence(client, database, picked)
    finally:
        await client.close()

    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    sheet = args.out / f"audit-{database}-seed{args.seed}-{stamp}.csv"
    _write_sheet(sheet, rows)
    manifest = {
        "database": database,
        "namespace": args.namespace,
        "seed": args.seed,
        "requested": args.size,
        "drawn": len(rows),
        "generated_at": datetime.now(UTC).isoformat(),
        "sheet": sheet.name,
        "tier_population": census,
        "tier_quota": quota,
        "predicate_population": dict(predicate_counts.most_common()),
    }
    sheet.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[audit] {database}: {len(edges)} current REL edges")
    for tier, population in census.items():
        print(f"  {tier:<13} population={population:<6} sampled={quota[tier]}")
    for predicate, count in predicate_counts.most_common(15):
        print(f"  {predicate:<13} population={count}")
    print(f"\n{len(predicate_counts)} distinct predicates")
    print(f"\nSheet     : {sheet}")
    print(f"Manifest  : {sheet.with_suffix('.json')}")
    print(
        f"\nFill the {', '.join(JUDGEMENT_COLUMNS)} columns with 1 or 0 "
        "(blank = not judged), then run:\n"
        f"  python -m evaluation audit report --sheet {sheet}"
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    rows = _read_sheet(args.sheet)
    if not rows:
        raise SystemExit(f"{args.sheet} has no rows")
    manifest_path = args.manifest or args.sheet.with_suffix(".json")
    manifest = _read_manifest(args.sheet, args.manifest)
    populations = (manifest or {}).get("tier_population")
    tiers = aggregate(rows, populations)
    report = build_report(
        tiers,
        {
            "sheet": args.sheet.name,
            "row": (manifest or {}).get("namespace") or (manifest or {}).get("database", ""),
            "database": (manifest or {}).get("database", "unknown"),
            "seed": (manifest or {}).get("seed", "unknown"),
            "generated_at": datetime.now(UTC).isoformat(),
            "rows": len(rows),
            "manifest": None if manifest is None else str(manifest_path),
        },
    )
    if manifest is None:
        report["caveats"].insert(
            0,
            "Sample manifest not found next to the sheet — tier populations and seed "
            "are unknown, so this report cannot say what fraction of each tier was sampled.",
        )

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise TypeError(f"baseline report {args.baseline} is not a JSON object")
        report["baseline"] = {"report": str(args.baseline), "deltas": compare(baseline, report)}

    out_dir = args.out or args.sheet.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.sheet.stem}-report.json"
    markdown_path = out_dir / f"{args.sheet.stem}-report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print(render_markdown(report))
    print(json_path)
    print(markdown_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation audit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sample = sub.add_parser("sample", help="Draw a stratified sheet of REL edges to judge.")
    sample.add_argument(
        "--namespace",
        default="",
        help="GraphKnows namespace to audit (its database is mem_<namespace>).",
    )
    sample.add_argument(
        "--database",
        default="",
        help="Raw ArcadeDB database name, when it does not follow the mem_<namespace> convention.",
    )
    sample.add_argument(
        "-n",
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            f"Edges to draw across all tiers (default {DEFAULT_SAMPLE_SIZE}). Spread over four "
            "tiers this is ~37 each, where a 95%% interval is roughly +/-15pp near p=0.5."
        ),
    )
    sample.add_argument("--seed", type=int, default=0, help="PRNG seed; same seed, same sheet.")
    sample.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")

    report = sub.add_parser("report", help="Turn a judged sheet into per-tier precision.")
    report.add_argument("--sheet", type=Path, required=True, help="The judged CSV.")
    report.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Sample manifest (default: the sheet's path with a .json suffix).",
    )
    report.add_argument(
        "--out", type=Path, default=None, help="Output directory (default: the sheet's directory)."
    )
    report.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="A previously generated report's JSON output to compare this run against.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    if args.mode == "sample":
        raise SystemExit(asyncio.run(_sample(args)))
    raise SystemExit(_report(args))


if __name__ == "__main__":
    main()
