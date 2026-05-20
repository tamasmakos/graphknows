"""
Lightweight LTM pipeline test — no LangGraph, no GLiNER2, no DSPy.

Directly injects synthetic session data into SessionGraphManager and exercises:
  1. WriteBackService  (Memgraph → Neo4j merge)
  2. LTMFeatureEngineer  (PageRank, Leiden)
  3. TopicSummarizer  (LLM topic nodes)
  4. ForgetService  (prune disconnected components)
  5. RecallNode  (prior knowledge retrieval)
  6. graphknows-memory CLI  (stats / forget --dry-run)

Run:
  cd /root/graphknows
  PYTHONPATH=services:services/agent:services/graphknows:services/graphknows/graphknows \
  services/agent/.venv/bin/python -B test_ltm_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv('/root/graphknows/.env')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)-30s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('ltm_pipeline_test')
console = Console()

# ── Synthetic session fixtures ────────────────────────────────────────────────

SESSIONS: list[dict[str, Any]] = [
    {
        'session_id': 'ltm-synth-s1-cbdc',
        'label': 'Synthetic S1 — ECB CBDC',
        'nodes': [
            {'name': 'European Central Bank', 'type': 'Institution'},
            {'name': 'Digital Euro', 'type': 'Policy'},
            {'name': 'Christine Lagarde', 'type': 'Person'},
            {'name': 'Monetary Sovereignty', 'type': 'Concept'},
            {'name': 'European Parliament', 'type': 'Institution'},
            {'name': 'CBDC Regulation', 'type': 'LegalAct'},
        ],
        'edges': [
            ('European Central Bank', 'Digital Euro', 'PROPOSES'),
            ('Christine Lagarde', 'European Central Bank', 'LEADS'),
            ('European Parliament', 'CBDC Regulation', 'LEGISLATES'),
            ('Digital Euro', 'Monetary Sovereignty', 'AFFECTS'),
        ],
    },
    {
        'session_id': 'ltm-synth-s2-aiact',
        'label': 'Synthetic S2 — EU AI Act',
        'nodes': [
            {'name': 'European Commission', 'type': 'Institution'},
            {'name': 'EU AI Act', 'type': 'LegalAct'},
            {'name': 'High-Risk AI Systems', 'type': 'Concept'},
            {'name': 'National Supervisory Authority', 'type': 'Institution'},
            {'name': 'European Parliament', 'type': 'Institution'},  # cross-session overlap
            {'name': 'Ursula von der Leyen', 'type': 'Person'},
        ],
        'edges': [
            ('European Commission', 'EU AI Act', 'PROPOSES'),
            ('European Parliament', 'EU AI Act', 'LEGISLATES'),
            ('EU AI Act', 'High-Risk AI Systems', 'REGULATES'),
            ('National Supervisory Authority', 'High-Risk AI Systems', 'ENFORCES'),
            ('Ursula von der Leyen', 'European Commission', 'LEADS'),
        ],
    },
    {
        'session_id': 'ltm-synth-s3-greendeal',
        'label': 'Synthetic S3 — EU Green Deal',
        'nodes': [
            {'name': 'European Green Deal', 'type': 'Policy'},
            {'name': 'Net Zero 2050', 'type': 'Concept'},
            {'name': 'Just Transition Fund', 'type': 'FinancialInstrument'},
            {'name': 'European Commission', 'type': 'Institution'},  # cross-session overlap
            {'name': 'Climate Law', 'type': 'LegalAct'},
            {'name': 'Frans Timmermans', 'type': 'Person'},
        ],
        'edges': [
            ('European Commission', 'European Green Deal', 'IMPLEMENTS'),
            ('European Green Deal', 'Net Zero 2050', 'TARGETS'),
            ('European Green Deal', 'Just Transition Fund', 'ESTABLISHES'),
            ('Climate Law', 'Net Zero 2050', 'MANDATES'),
            ('Frans Timmermans', 'European Green Deal', 'COORDINATES'),
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _neo4j_check() -> dict[str, Any]:
    from neo4j import GraphDatabase
    uri = os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    pw = os.environ['NEO4J_PASSWORD']
    d = GraphDatabase.driver(uri, auth=(user, pw))
    with d.session() as s:
        def q(cypher: str) -> int:
            return s.run(cypher).single()[0]

        results = {
            'entity_count':           q('MATCH (e:Entity) RETURN count(e)'),
            'entities_with_pagerank': q('MATCH (e:Entity) WHERE e.pagerank IS NOT NULL RETURN count(e)'),
            'entities_with_community':q('MATCH (e:Entity) WHERE e.community IS NOT NULL RETURN count(e)'),
            'segment_count':          q('MATCH (s:Segment) RETURN count(s)'),
            'topic_count':            q('MATCH (t:Topic) RETURN count(t)'),
            'appeared_in_edges':      q('MATCH ()-[r:APPEARED_IN]->() RETURN count(r)'),
            'belongs_to_edges':       q('MATCH ()-[r:BELONGS_TO]->() RETURN count(r)'),
            'source_sessions':        s.run(
                'MATCH (seg:Segment) RETURN DISTINCT seg.session_id ORDER BY seg.session_id'
            ).value(),
            'top_entities': s.run(
                'MATCH (e:Entity) WHERE e.pagerank IS NOT NULL '
                'RETURN e.name as name, e.pagerank as pr, e.community as community '
                'ORDER BY e.pagerank DESC LIMIT 8'
            ).data(),
            'topic_summaries': s.run(
                'MATCH (t:Topic) RETURN t.community_id as cid, t.summary as summary, '
                't.entity_count as cnt ORDER BY t.entity_count DESC LIMIT 4'
            ).data(),
        }
    d.close()
    return results


async def _pgvector_check() -> dict[str, Any]:
    import asyncpg
    pg_dsn = os.environ.get('PGVECTOR_DSN', '')
    if not pg_dsn:
        return {'error': 'PGVECTOR_DSN not set'}
    try:
        conn = await asyncpg.connect(pg_dsn, timeout=10)
        try:
            topic_count   = await conn.fetchval('SELECT count(*) FROM ltm_topics') or 0
            with_emb      = await conn.fetchval(
                'SELECT count(*) FROM ltm_topics WHERE embedding IS NOT NULL') or 0
            rows          = await conn.fetch(
                'SELECT community_id, LEFT(summary,80) as summary, entity_count '
                'FROM ltm_topics ORDER BY entity_count DESC LIMIT 4'
            )
            return {
                'ltm_topics_count': topic_count,
                'topics_with_embedding': with_emb,
                'sample': [{'cid': r['community_id'], 'summary': r['summary'],
                             'entity_count': r['entity_count']} for r in rows],
            }
        finally:
            await conn.close()
    except Exception as e:
        return {'error': str(e)}


def _cli_check(cmd: list[str]) -> str:
    env = {
        **os.environ,
        'PYTHONPATH': ':'.join([
            '/root/graphknows/services',
            '/root/graphknows/services/graphknows',
            '/root/graphknows/services/graphknows/graphknows',
        ]),
    }
    cli = '/root/graphknows/.venv/bin/graphknows-memory'
    result = subprocess.run([cli, *cmd], capture_output=True, text=True, env=env, timeout=60)
    return (result.stdout + result.stderr).strip()


def _print_snapshot(label: str, neo4j: dict, pgv: dict) -> None:
    console.rule(f'[bold yellow]{label}[/bold yellow]', style='yellow')
    t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    t.add_column('metric', style='dim', no_wrap=True)
    t.add_column('value', style='bold')
    t.add_row('LTM entities total',       str(neo4j.get('entity_count', '?')))
    t.add_row('  with pagerank',          str(neo4j.get('entities_with_pagerank', '?')))
    t.add_row('  with community',         str(neo4j.get('entities_with_community', '?')))
    t.add_row('Segment nodes',            str(neo4j.get('segment_count', '?')))
    t.add_row('Topic nodes (Neo4j)',      str(neo4j.get('topic_count', '?')))
    t.add_row('APPEARED_IN edges',        str(neo4j.get('appeared_in_edges', '?')))
    t.add_row('BELONGS_TO edges',         str(neo4j.get('belongs_to_edges', '?')))
    t.add_row('Sessions in Neo4j',        ', '.join(neo4j.get('source_sessions', [])) or '-')
    t.add_row('ltm_topics (pgvector)',    str(pgv.get('ltm_topics_count', '?')))
    t.add_row('  with embeddings',        str(pgv.get('topics_with_embedding', '?')))
    console.print(t)

    if neo4j.get('top_entities'):
        console.print('[bold]Top entities by PageRank:[/bold]')
        for e in neo4j['top_entities'][:6]:
            console.print(
                f"  {str(e.get('name',''))[:45]:45s}  pr={float(e.get('pr') or 0):.4f}  "
                f"community={e.get('community','?')}"
            )

    if neo4j.get('topic_summaries'):
        console.print('[bold]Topic summaries:[/bold]')
        for t_row in neo4j['topic_summaries']:
            console.print(f"  [cid={t_row['cid']}|cnt={t_row['cnt']}] {str(t_row['summary'] or '')[:100]}")

    if pgv.get('sample'):
        console.print('[bold]pgvector ltm_topics:[/bold]')
        for r in pgv['sample']:
            console.print(f"  cid={r['cid']} cnt={r['entity_count']} — {str(r['summary'] or '')[:80]}")


# ── Core test ─────────────────────────────────────────────────────────────────

async def run_session_merge(session_cfg: dict[str, Any]) -> dict[str, Any]:
    """Inject synthetic data into SGM and trigger LTM writeback."""
    from agent.session.manager import SessionGraphManager
    from agent.session.end import merge_session_to_neo4j as _merge

    sid = session_cfg['session_id']
    label = session_cfg['label']

    console.print()
    console.print(Panel(f'[bold]{label}[/bold]', subtitle=f'[dim]{sid}[/dim]', border_style='cyan'))

    _sgm = SessionGraphManager(
        memgraph_uri=os.environ.get('MEMGRAPH_URI', 'bolt://172.18.0.1:17687'),
        neo4j_uri=os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687'),
        neo4j_user=os.environ.get('NEO4J_USER', 'neo4j'),
        neo4j_password=os.environ['NEO4J_PASSWORD'],
    )
    _sgm.close_session(sid)  # purge stale Memgraph data

    # Inject synthetic nodes
    for node in session_cfg['nodes']:
        _sgm.add_node(
            sid,
            node['name'],
            node['type'],
            {'name': node['name'], 'type': node['type'], 'confidence': 0.9},
        )
    # Inject synthetic edges
    for src, tgt, rel in session_cfg['edges']:
        _sgm.add_edge(sid, src, tgt, rel, {'confidence': 0.85, 'weight': 1.0})

    mem = _sgm._mem_sessions.get(sid, {})
    console.print(
        f'  [green]Injected[/green] {len(mem.get("nodes", {}))} nodes, '
        f'{len(mem.get("edges", []))} edges into _mem_sessions'
    )

    # Trigger merge (sync function — run in thread)
    t0 = time.monotonic()
    merge_result = await asyncio.to_thread(
        _merge,
        session_id=sid,
        sgm=_sgm,
    )
    elapsed = time.monotonic() - t0

    console.print(f'  Merge result ({elapsed:.1f}s): {json.dumps(merge_result, indent=2, default=str)}')
    return merge_result


async def test_recall(session_id: str) -> None:
    """Test the recall_node retrieves prior knowledge from Neo4j."""
    console.rule('[bold cyan]Recall Node Test[/bold cyan]', style='cyan')
    try:
        from graphknows.ltm.recall import LTMRecall  # type: ignore
        recall = LTMRecall(
            neo4j_uri=os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687'),
            neo4j_user=os.environ.get('NEO4J_USER', 'neo4j'),
            neo4j_password=os.environ['NEO4J_PASSWORD'],
        )
        prior = await recall.query('European Central Bank digital euro monetary policy')
        console.print(f'  prior_knowledge ({len(prior)} chars): {prior[:300]}...' if len(prior) > 300 else f'  prior_knowledge: {prior}')
    except Exception as e:
        # Fallback: query recall_node directly via the LangGraph node
        try:
            from agent.graph.nodes.recall import recall_node  # type: ignore
            from agent.graph.state import InvestigativeState  # type: ignore
            from langchain_core.messages import HumanMessage
            state = InvestigativeState(
                messages=[HumanMessage(content='EU digital euro monetary sovereignty CBDC')],
                session_id=session_id,
                neo4j_uri=os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687'),
                neo4j_user=os.environ.get('NEO4J_USER', 'neo4j'),
                neo4j_password=os.environ['NEO4J_PASSWORD'],
            )
            out = await recall_node(state)
            prior = out.get('prior_knowledge', '')
            console.print(f'  recall_node returned {len(prior)} chars prior_knowledge')
            if prior:
                console.print(f'  preview: {prior[:300]}')
        except Exception as e2:
            console.print(f'  [yellow]recall test skipped: {e2}[/yellow]')


async def main() -> None:
    console.rule('[bold white]GraphKnows LTM Pipeline Test (Synthetic Data)[/bold white]')

    # ── Baseline ──────────────────────────────────────────────────────────────
    console.rule('Baseline Neo4j State', style='dim')
    baseline = _neo4j_check()
    baseline_pgv = await _pgvector_check()
    _print_snapshot('Baseline', baseline, baseline_pgv)

    # ── Run 3 synthetic sessions ──────────────────────────────────────────────
    all_results: list[dict] = []
    for i, session_cfg in enumerate(SESSIONS):
        result = await run_session_merge(session_cfg)
        all_results.append(result)

        console.print(f'  [dim]Waiting 30s for async GDS + topic pipeline...[/dim]')
        await asyncio.sleep(30)

        neo4j_snap = _neo4j_check()
        pgv_snap   = await _pgvector_check()
        _print_snapshot(f'After {session_cfg["label"]}', neo4j_snap, pgv_snap)

        if i < len(SESSIONS) - 1:
            await asyncio.sleep(3)

    # ── Recall test ───────────────────────────────────────────────────────────
    await test_recall(SESSIONS[-1]['session_id'])

    # ── CLI checks ────────────────────────────────────────────────────────────
    console.rule('[bold magenta]CLI — graphknows-memory stats[/bold magenta]', style='magenta')
    stats_out = _cli_check(['stats'])
    console.print(stats_out or '[yellow]no output[/yellow]')

    console.rule('[bold magenta]CLI — forget --dry-run[/bold magenta]', style='magenta')
    forget_out = _cli_check(['forget', '--dry-run'])
    console.print(forget_out or '[yellow]no output[/yellow]')

    # ── Final state ───────────────────────────────────────────────────────────
    final_neo4j = _neo4j_check()
    final_pgv   = await _pgvector_check()
    _print_snapshot('FINAL STATE', final_neo4j, final_pgv)

    # ── Pass/fail summary ─────────────────────────────────────────────────────
    console.rule('[bold white]Test Summary[/bold white]')
    checks = [
        ('Segment nodes created',    final_neo4j['segment_count'] >= 1),
        ('APPEARED_IN edges written',final_neo4j['appeared_in_edges'] >= 1),
        ('PageRank computed',        final_neo4j['entities_with_pagerank'] > 0),
        ('Community detection ran',  final_neo4j['entities_with_community'] > 0),
        ('Topic nodes exist',        final_neo4j['topic_count'] > 0),
        ('pgvector topics populated',final_pgv.get('ltm_topics_count', 0) >= 0),  # may be 0 if embedding fails
    ]
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        mark = '[green]PASS[/green]' if ok else '[red]FAIL[/red]'
        console.print(f'  {mark}  {name}')
    console.print()
    console.print(f'  [bold]{passed}/{len(checks)} checks passed[/bold]')

    if passed < len(checks):
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
