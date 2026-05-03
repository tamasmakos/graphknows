"""
run.py — CLI entry point for GraphKnows GraphRAG.

Usage:
  python run.py query "What did EU leaders say about Rule of Law?"
  python run.py ingest
  python run.py ingest --chunks /path/to/lkif_chunks.json --no-summarize
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def _header():
    console.print(Panel.fit(
        "[bold cyan]GraphKnows[/] [dim]Political Intelligence GraphRAG[/]",
        border_style="cyan", padding=(0, 2)
    ))


def cmd_ingest(args):
    from graph_store import load_env
    from ingest import ingest

    cfg = load_env()
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        console.print(f"[red]Chunks file not found:[/] {chunks_path}")
        sys.exit(1)

    _header()
    console.print(f"\n[bold]Ingesting[/] {chunks_path.name} ...\n")

    result = ingest(chunks_path, cfg, summarize=not args.no_summarize)

    t = Table(title="Ingestion Results", box=box.ROUNDED, border_style="green")
    t.add_column("Store", style="bold")
    t.add_column("Count", justify="right", style="cyan")
    t.add_row("Neo4j Chunks",   str(result.get("chunks", "?")))
    t.add_row("Neo4j Entities", str(result.get("entities", "?")))
    t.add_row("Neo4j Topics",   str(result.get("topics", "?")))
    t.add_row("Neo4j MENTIONS", str(result.get("mentions", "?")))
    t.add_row("pgvector Chunks",str(result.get("pgvector_chunks", "?")))
    console.print(t)


def cmd_query(args):
    import logging
    logging.basicConfig(level=logging.WARNING)  # suppress INFO during query display

    from graph_store import load_env
    from agent import GraphRAGAgent

    cfg = load_env()
    _header()
    console.print(f"\n[bold yellow]Query:[/] {args.query}\n")

    agent = GraphRAGAgent(cfg)
    result = agent.run(args.query, max_steps=args.steps)
    agent.close()

    # ── Reasoning Trace ──────────────────────────────────────────────────
    console.print(Panel("[bold]Reasoning Trace[/]", border_style="blue", expand=False))

    for step in result["trace"]:
        step_num = step["step"]
        thought = step.get("thought") or ""
        action  = step.get("action")
        obs     = step.get("observation")
        is_final = step.get("final", False)

        console.print(f"\n[bold blue]Step {step_num}[/]")
        if thought:
            console.print(f"  [dim]Thought:[/] {thought}")
        if action:
            console.print(f"  [bold green]Action:[/] [green]{action}[/]")
        if obs is not None and not is_final:
            obs_str = json.dumps(obs, ensure_ascii=False)
            preview = obs_str[:300] + ("..." if len(obs_str) > 300 else "")
            console.print(f"  [dim]Observation:[/] {preview}")
        if is_final:
            console.print(f"  [bold yellow]→ Final answer synthesized[/]")

    # ── Final Answer ──────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text(result["answer"], style="white"),
        title="[bold cyan]Answer[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Citations ─────────────────────────────────────────────────────────
    if result["citations"]:
        console.print()
        t = Table(title="Source Citations", box=box.SIMPLE, border_style="dim")
        t.add_column("Chunk", style="bold cyan", no_wrap=True)
        t.add_column("Document", style="dim")
        t.add_column("Excerpt")
        for c in result["citations"]:
            t.add_row(c["chunk_id"], c.get("doc_id", "?"), c.get("excerpt", "")[:120] + "...")
        console.print(t)

    # ── Stats ─────────────────────────────────────────────────────────────
    console.print(f"\n[dim]Completed in {result['steps']} reasoning steps.[/]")


def cmd_stats(args):
    from graph_store import load_env, NeoStore, PgStore
    cfg = load_env()
    _header()

    neo = NeoStore(cfg["neo4j_uri"], cfg["neo4j_user"], cfg["neo4j_password"])
    s = neo.stats()
    neo.close()

    pg = PgStore(cfg["pgvector_dsn"])
    cur = pg.conn.cursor()
    cur.execute("SELECT count(*) FROM chunks")
    pg_chunks = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM entities")
    pg_ents = cur.fetchone()[0]
    pg.close()

    t = Table(title="Graph Store Stats", box=box.ROUNDED, border_style="cyan")
    t.add_column("Store", style="bold")
    t.add_column("Key", style="dim")
    t.add_column("Value", justify="right", style="cyan")
    t.add_row("Neo4j", "Chunks",   str(s.get("chunks", 0)))
    t.add_row("Neo4j", "Entities", str(s.get("entities", 0)))
    t.add_row("Neo4j", "Topics",   str(s.get("topics", 0)))
    t.add_row("Neo4j", "MENTIONS", str(s.get("mentions", 0)))
    t.add_row("pgvector", "Chunks",   str(pg_chunks))
    t.add_row("pgvector", "Entities", str(pg_ents))
    console.print(t)


def main():
    parser = argparse.ArgumentParser(prog="graphknows", description="Political Intelligence GraphRAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest LKIF chunks into Neo4j + pgvector")
    p_ingest.add_argument("--chunks", default=str(ROOT / "output_political_smoke" / "lkif_chunks.json"))
    p_ingest.add_argument("--no-summarize", action="store_true", help="Skip LLM community summaries")

    # query
    p_query = sub.add_parser("query", help="Run a GraphRAG query")
    p_query.add_argument("query", help="Your question about EU politics")
    p_query.add_argument("--steps", type=int, default=8, help="Max reasoning steps")

    # stats
    sub.add_parser("stats", help="Show graph store statistics")

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
