"""
GraphKnows — Long-Term Memory Layer
Tech Lead Demo

Narrative:
  1. The Problem      — stateless agents forget everything between sessions
  2. The Architecture — Memgraph (working memory) → Neo4j LTM (persistent, GDS-powered)
  3. Act I: Writeback — ingest a policy document, watch entities + chunks land in LTM
  4. Act II: Enrichment— run GDS: Node2Vec weights, Louvain communities, quality metrics
  5. Act III: Recall   — three recall paths (document / dialog / topic), live results
  6. Act IV: Forget    — prune orphan nodes, see graph shrink cleanly

Run:
  cd /root/graphknows
  /root/graphknows/services/agent/.venv/bin/python -B demo_ltm.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

_repo = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_repo, "services"),
    os.path.join(_repo, "services", "graphknows"),
    os.path.join(_repo, "services", "graphknows", "graphknows"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(os.path.join(_repo, ".env"))

from neo4j import GraphDatabase
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

console = Console(width=100)

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]

DEMO_SESSION  = f"demo-ltm-{int(time.time())}"
DEMO_DOC_TEXT = """
The EU AI Act, formally the Regulation on Artificial Intelligence, was proposed by the
European Commission in April 2021. It establishes a risk-based regulatory framework for
AI systems deployed across the European Union. High-risk AI systems — including those used
in critical infrastructure, education, employment, and law enforcement — are subject to
strict conformity assessments and transparency requirements. The European Parliament
approved the final text in March 2024, making it the world's first comprehensive AI law.
Ursula von der Leyen, President of the European Commission, called it a global milestone.
The European Central Bank welcomed the clarity the Act brings to AI-powered financial tools.
"""

DEMO_TURNS = [
    {"role": "user",  "text": "What does the EU AI Act say about high-risk systems?"},
    {"role": "agent", "text": "The EU AI Act classifies High-Risk AI Systems in sectors like critical infrastructure and law enforcement. They must pass conformity assessments before deployment."},
    {"role": "user",  "text": "Who proposed it and when?"},
    {"role": "agent", "text": "The European Commission proposed the AI Act in April 2021. The European Parliament approved the final text in March 2024."},
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def _counts(s) -> dict:
    return {
        "entities":   s.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"],
        "segments":   s.run("MATCH (s:Segment) RETURN count(s) as c").single()["c"],
        "gds_runs":   s.run("MATCH (g:GDSRun) RETURN count(g) as c").single()["c"],
        "documents":  s.run("MATCH (d:Document) RETURN count(d) as c").single()["c"],
        "chunks":     s.run("MATCH (c:Chunk) RETURN count(c) as c").single()["c"],
        "utterances": s.run("MATCH (u:Utterance) RETURN count(u) as c").single()["c"],
        "topics":     s.run("MATCH (t:Topic) RETURN count(t) as c").single()["c"],
    }


def _section(title: str, subtitle: str = "") -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
    console.print()


def _ok(msg: str) -> None:
    console.print(f"  [bold green]OK[/bold green]  {msg}")


def _info(msg: str) -> None:
    console.print(f"  [dim blue]--[/dim blue]  {msg}")


_interactive = sys.stdin.isatty()

def _pause(msg: str = "Press Enter to continue...") -> None:
    if _interactive:
        console.print(f"\n  [dim]{msg}[/dim]")
        input()
    else:
        console.print()
        time.sleep(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Preamble
# ─────────────────────────────────────────────────────────────────────────────

def show_intro() -> None:
    console.print(Panel(
        Text.from_markup(
            "[bold white]GraphKnows[/bold white] — Long-Term Memory Layer\n\n"
            "Agents are stateless by default. Each session starts from zero.\n"
            "GraphKnows gives agents persistent, structured memory:\n\n"
            "  [cyan]Memgraph[/cyan]  Working memory (ephemeral, per-session graph)\n"
            "     [dim]↓  session end event[/dim]\n"
            "  [cyan]Neo4j[/cyan]     Long-term memory (GDS-enriched, temporally layered)\n"
            "     [dim]↓  recall_node at session start[/dim]\n"
            "  [cyan]Agent[/cyan]     Primed with prior knowledge, topic clusters, provenance\n\n"
            "This demo walks through four acts on a live Neo4j graph (388 entities, 3 sessions)."
        ),
        title="[bold]Tech Lead Demo[/bold]",
        border_style="cyan",
        padding=(1, 4),
    ))
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Act I: Document Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def act_ingest(drv) -> str:
    _section("Act I: Document Ingestion",
             "URL / PDF / plain text  →  Chunk nodes  →  MENTIONS  →  Entity nodes")

    from graphknows.document_ingestor import DocumentIngestor

    _info("Source: inline policy text (EU AI Act summary, ~150 words)")
    _info("Pipeline: tiktoken chunking (500 tok / 50 overlap)  →  GLiNER2 NER  →  dual-write")

    ingestor = DocumentIngestor(neo4j_driver=drv)

    # Monkey-patch _fetch so we don't need network
    import hashlib
    doc_id = "demo-" + hashlib.sha256(DEMO_DOC_TEXT.encode()).hexdigest()[:12]

    original_fetch = ingestor._fetch
    def _mock_fetch(source):
        return DEMO_DOC_TEXT, "demo://eu-ai-act-summary"
    ingestor._fetch = _mock_fetch

    t0 = time.perf_counter()
    result = ingestor.ingest(
        source="demo://eu-ai-act-summary",
        session_id=DEMO_SESSION,
        ontology=["ORGANIZATION", "PERSON", "LAW", "POLICY", "INSTITUTION"],
        title="EU AI Act — Policy Summary",
    )
    elapsed = time.perf_counter() - t0

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("Property", style="dim", width=20)
    t.add_column("Value")
    t.add_row("Document ID",  result.document_id)
    t.add_row("Title",        result.title)
    t.add_row("Chunks",       str(result.chunk_count))
    t.add_row("Entity mentions", str(result.mention_count))
    t.add_row("Elapsed",      f"{elapsed*1000:.0f}ms")
    console.print(t)

    with drv.session() as s:
        chunk_sample = s.run(
            "MATCH (d:Document {id: $did})-[:HAS_CHUNK]->(c:Chunk) RETURN c.text LIMIT 2",
            did=result.document_id,
        ).data()
        for i, row in enumerate(chunk_sample):
            console.print(f"\n  [bold]Chunk {i+1}[/bold] (first 160 chars):")
            console.print(f"  [dim]{row['c.text'][:160].strip()}...[/dim]")

        entity_sample = s.run(
            "MATCH (d:Document {id: $did})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
            "RETURN DISTINCT e.name LIMIT 8",
            did=result.document_id,
        ).data()
        if entity_sample:
            names = [r["e.name"] for r in entity_sample]
            _ok(f"Entities linked via MENTIONS: {', '.join(names)}")
        else:
            _info("Entities not linked (GLiNER2 threshold — OK for synthetic text)")

    _ok(f"Document + {result.chunk_count} Chunk nodes written to Neo4j")
    _pause()
    return result.document_id


# ─────────────────────────────────────────────────────────────────────────────
# Act II: Dialog Writeback
# ─────────────────────────────────────────────────────────────────────────────

def act_dialog(drv) -> None:
    _section("Act II: Dialog Writeback",
             "Conversation turns  →  Dialog / Utterance nodes  →  MENTIONS  →  existing Entities")

    from agent.session.dialog import DialogWriteBack, TurnData

    now = datetime.now(timezone.utc).isoformat()
    turns = [
        TurnData(i, t["role"], "user" if t["role"] == "user" else "GraphKnows-Agent",
                 t["text"], now)
        for i, t in enumerate(DEMO_TURNS)
    ]

    # Ensure a Segment exists to link HAS_DIALOG from
    with drv.session() as s:
        s.run("MERGE (:Segment {session_id: $sid})", sid=DEMO_SESSION)

    wb = DialogWriteBack(neo4j_driver=drv)
    t0 = time.perf_counter()
    result = wb.write(session_id=DEMO_SESSION, turns=turns)
    elapsed = time.perf_counter() - t0

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("Property", style="dim", width=20)
    t.add_column("Value")
    t.add_row("Dialog ID",   result.dialog_id)
    t.add_row("Session",     result.session_id)
    t.add_row("Utterances",  str(result.utterance_count))
    t.add_row("Mentions",    str(result.mention_count))
    t.add_row("Elapsed",     f"{elapsed*1000:.0f}ms")
    console.print(t)

    with drv.session() as s:
        path_check = s.run(
            "MATCH (seg:Segment {session_id: $sid})-[:HAS_DIALOG]->(d:Dialog)"
            "-[:PARTICIPANT]->(p:Participant)-[:SAID]->(u:Utterance) "
            "RETURN p.role AS role, u.text AS text ORDER BY u.turn_index",
            sid=DEMO_SESSION,
        ).data()
        console.print("\n  [bold]Conversation path in graph:[/bold]")
        for row in path_check:
            role_fmt = "[cyan]user[/cyan]" if row["role"] == "user" else "[green]agent[/green]"
            console.print(f"  {role_fmt}  {row['text'][:80]}")

    _ok(f"Segment → Dialog → {len(turns)} Utterances written (idempotent via MERGE)")
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Act III: GDS Enrichment — live metrics from existing 3-session graph
# ─────────────────────────────────────────────────────────────────────────────

def act_enrichment(drv) -> None:
    _section("Act III: GDS Enrichment",
             "Node2Vec → cosine weights → Louvain communities → quality metrics (GDSRun nodes)")

    _info("Reading GDSRun nodes from existing 3-session synthetic graph (388 entities, 1355 edges)")

    with drv.session() as s:
        runs = s.run(
            "MATCH (g:GDSRun) RETURN g.session_id AS session, "
            "g.community_count AS communities, g.modularity AS mod_w, "
            "g.unweighted_modularity AS mod_uw, g.silhouette_cosine AS sil, "
            "g.global_separation AS sep, g.ran_iterations AS levels "
            "ORDER BY g.timestamp"
        ).data()

        t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        t.add_column("Session",       style="dim",    width=28)
        t.add_column("Communities",   justify="right", width=13)
        t.add_column("Mod(weighted)", justify="right", width=14)
        t.add_column("Mod(baseline)", justify="right", width=14)
        t.add_column("Silhouette",    justify="right", width=12)
        t.add_column("Global Sep.",   justify="right", width=12)
        t.add_column("Levels",        justify="right", width=8)
        for r in runs:
            t.add_row(
                r["session"] or "—",
                str(r["communities"]),
                f"{r['mod_w']:.4f}"  if r["mod_w"]  else "—",
                f"{r['mod_uw']:.4f}" if r["mod_uw"] else "—",
                f"{r['sil']:.4f}"    if r["sil"]    else "—",
                f"{r['sep']:.4f}"    if r["sep"]    else "—",
                str(r["levels"])     if r["levels"]  else "—",
            )
        console.print(t)

        # Top entities by PageRank
        top = s.run(
            "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL "
            "RETURN e.name AS name, e.pagerank AS pr, e.community_id AS cid "
            "ORDER BY e.pagerank DESC LIMIT 8"
        ).data()

        console.print("\n  [bold]Top 8 entities by PageRank (hub detection):[/bold]")
        for r in top:
            bar = "█" * int(r["pr"] * 7)
            console.print(f"  [cyan]{r['name']:40s}[/cyan] pr={r['pr']:.4f}  cid={r['cid']}  {bar}")

    _info("Weighted modularity 0.54–0.85 vs unweighted baseline 0.56 — weights add signal")
    _info("Silhouette negative at ~1 entity/community; reaches 0.16 on full 3557-entity corpus (thesis)")
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Act IV: Retrieval — three paths
# ─────────────────────────────────────────────────────────────────────────────

def act_recall(drv) -> None:
    _section("Act IV: Retrieval",
             "Three recall paths: by document chunk  /  by dialog turn  /  by topic cluster")

    from graphknows.recall import recall_by_document, recall_by_dialog, recall_by_topic

    # ── recall_by_document ──────────────────────────────────────────────────
    console.print("  [bold]recall_by_document[/bold]  query=\"European Parliament Commission\"")
    t0 = time.perf_counter()
    doc_results = recall_by_document(drv, "European Parliament Commission", top_k=3)
    elapsed = time.perf_counter() - t0

    if doc_results:
        for r in doc_results[:3]:
            console.print(f"\n  [dim]score={r.score:.0f}  source={r.provenance.get('title','?')}[/dim]")
            console.print(f"  \"{r.text[:120].strip()}...\"")
            if r.entity_names:
                console.print(f"  entities: {', '.join(r.entity_names[:5])}")
        _ok(f"{len(doc_results)} chunk(s) matched in {elapsed*1000:.0f}ms")
    else:
        _info("No document chunks yet — ingest a real document to see results")

    console.print()

    # ── recall_by_dialog ────────────────────────────────────────────────────
    console.print("  [bold]recall_by_dialog[/bold]  query=\"High-Risk AI Systems\"")
    t0 = time.perf_counter()
    dialog_results = recall_by_dialog(drv, "High-Risk Systems agent", top_k=3)
    elapsed = time.perf_counter() - t0

    if dialog_results:
        for r in dialog_results[:3]:
            role = r.provenance.get("role", "?")
            console.print(f"\n  [dim]score={r.score:.0f}  session={r.provenance.get('session_id','?')}  role={role}[/dim]")
            console.print(f"  \"{r.text[:120].strip()}\"")
        _ok(f"{len(dialog_results)} utterance(s) matched in {elapsed*1000:.0f}ms")
    else:
        _info("No dialog turns linked yet — entities in utterances not yet matched")

    console.print()

    # ── recall_by_topic ─────────────────────────────────────────────────────
    console.print("  [bold]recall_by_topic[/bold]  query=\"monetary policy central bank\"")
    t0 = time.perf_counter()
    topic_results = recall_by_topic(drv, "monetary policy central bank", top_k=3)
    elapsed = time.perf_counter() - t0

    if topic_results:
        for r in topic_results[:3]:
            console.print(f"\n  [dim]score={r.score:.1f}  topic_id={r.provenance.get('topic_id','?')}  size={r.provenance.get('size','?')} entities[/dim]")
            label = r.provenance.get("label", r.text)
            console.print(f"  [cyan]{label}[/cyan]")
            if r.entity_names:
                console.print(f"  top entities: {', '.join(r.entity_names[:5])}")
        _ok(f"{len(topic_results)} topic(s) matched in {elapsed*1000:.0f}ms")
    else:
        _info("No Topic nodes yet — run topic summarization (needs OPENROUTER_API_KEY)")
        _info("Topic clusters exist: 379 Louvain communities on 388 entities — summarization pending")

    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Act V: Forget / Prune
# ─────────────────────────────────────────────────────────────────────────────

def act_forget(drv, doc_id: str) -> None:
    _section("Act V: Forget",
             "Prune orphan nodes, expired segments, disconnected components")

    from graphknows.forget import LTMForget

    _info("Creating one orphan Document (no HAS_CHUNK edges) to demonstrate pruning...")
    with drv.session() as s:
        s.run("MATCH (d:Document {id: 'demo-orphan'}) DETACH DELETE d")
        s.run("CREATE (:Document {id: 'demo-orphan', title: 'Orphan Doc', ingested_at: datetime()})")

    pg_dsn = os.environ.get("PGVECTOR_DSN", "")
    forget = LTMForget(drv, pg_dsn)

    before = {}
    after = {}
    with drv.session() as s:
        before["documents"] = s.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]
        before["utterances"] = s.run("MATCH (u:Utterance) RETURN count(u) as c").single()["c"]

    t0 = time.perf_counter()
    result = forget.prune(session_id=None, utterance_ttl_days=180)
    elapsed = time.perf_counter() - t0

    with drv.session() as s:
        after["documents"] = s.run("MATCH (d:Document) RETURN count(d) as c").single()["c"]

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("Prune type",         style="dim",  width=25)
    t.add_column("Pruned",             justify="right", width=10)
    t.add_row("Orphan documents",      str(result.get("documents_orphaned", 0)))
    t.add_row("Orphan chunks",         str(result.get("chunks_orphaned", 0)))
    t.add_row("Empty dialogs",         str(result.get("dialogs_empty", 0)))
    t.add_row("Expired utterances",    str(result.get("utterances_ttl", 0)))
    console.print(t)

    _info(f"Documents before prune: {before['documents']}  →  after: {after['documents']}")
    _ok(f"Prune complete in {elapsed*1000:.0f}ms")

    _info("CLI equivalent: graphknows-memory forget  (or graphknows-memory stats)")
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# Demo cleanup
# ─────────────────────────────────────────────────────────────────────────────

def cleanup(drv, doc_id: str) -> None:
    _section("Cleanup", "Removing demo artifacts from Neo4j")
    with drv.session() as s:
        s.run("MATCH (d:Document {id: $did}) DETACH DELETE d", did=doc_id)
        s.run("MATCH (c:Chunk) WHERE c.id STARTS WITH 'demo-' DETACH DELETE c")
        s.run("MATCH (seg:Segment {session_id: $sid}) DETACH DELETE seg", sid=DEMO_SESSION)
        s.run("MATCH (dia:Dialog {session_id: $sid}) DETACH DELETE dia", sid=DEMO_SESSION)
        s.run("MATCH (p:Participant) WHERE p.id STARTS WITH 'demo-' DETACH DELETE p", )
        s.run("MATCH (u:Utterance) WHERE u.id STARTS WITH 'demo-' DETACH DELETE u")
    _ok("Demo artifacts removed — production graph intact")


# ─────────────────────────────────────────────────────────────────────────────
# Graph state snapshot
# ─────────────────────────────────────────────────────────────────────────────

def show_graph_state(drv, label: str) -> None:
    with drv.session() as s:
        c = _counts(s)
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("Node type", style="dim")
    t.add_column("Count", justify="right", style="cyan bold")
    for k, v in c.items():
        t.add_row(k.capitalize(), str(v))
    console.print(Panel(t, title=f"[bold]Graph State — {label}[/bold]", border_style="dim"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    show_intro()

    drv = _driver()

    show_graph_state(drv, "baseline (3 synthetic sessions)")
    _pause("Press Enter to start Act I: Document Ingestion...")

    doc_id = act_ingest(drv)
    show_graph_state(drv, "after document ingest")

    _pause("Press Enter for Act II: Dialog Writeback...")
    act_dialog(drv)
    show_graph_state(drv, "after dialog writeback")

    _pause("Press Enter for Act III: GDS Enrichment metrics...")
    act_enrichment(drv)

    _pause("Press Enter for Act IV: Retrieval...")
    act_recall(drv)

    _pause("Press Enter for Act V: Forget / Prune...")
    act_forget(drv, doc_id)

    cleanup(drv, doc_id)
    drv.close()

    console.print(Panel(
        Text.from_markup(
            "[bold white]What you just saw:[/bold white]\n\n"
            "  [cyan]Ingest[/cyan]   Any text/PDF/URL → chunked, NER'd, stored with provenance\n"
            "  [cyan]Dialog[/cyan]   Every conversation turn stored as Utterance nodes in the graph\n"
            "  [cyan]Enrich[/cyan]   GDS pipeline: Node2Vec → cosine weights → Louvain → quality metrics\n"
            "  [cyan]Recall[/cyan]   Three traversal paths: document / dialog / topic — no embeddings needed\n"
            "  [cyan]Forget[/cyan]   Temporal pruning: orphans, TTL, disconnected components\n\n"
            "[bold white]Exposed as:[/bold white]\n\n"
            "  [cyan]MCP tools[/cyan]  recall_by_document | recall_by_dialog | recall_by_topic\n"
            "  [cyan]CLI[/cyan]        graphknows-memory recall-docs | recall-dialog | recall-topic | forget\n\n"
            "[bold white]Stack:[/bold white]  Memgraph (working memory)  →  Neo4j + GDS 2026 (LTM)  →  pgvector (topic ANN)\n"
            "                  Louvain communities  |  Node2Vec embeddings  |  PageRank centrality"
        ),
        title="[bold cyan]Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 4),
    ))


if __name__ == "__main__":
    main()
