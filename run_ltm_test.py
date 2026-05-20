"""Multi-session LTM integration test harness.

Runs 3 sequential investigations on related EU policy topics, then validates
every LTM feature after each session:

  Session 1 — ECB digital euro / CBDC
  Session 2 — EU AI Act (should recall digital governance context)
  Session 3 — EU Green Deal financing (should recall both prior sessions)

Verification checks after all 3 sessions:
  - Segment nodes in Neo4j
  - Entity nodes enriched with pagerank + community_id
  - Topic nodes with summaries in Neo4j + pgvector ltm_topics
  - Recall ANN search returns relevant results
  - graphknows-memory CLI: stats / query / forget --help
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, '/root/graphknows/services')
sys.path.insert(0, '/root/graphknows/services/agent')
sys.path.insert(0, '/root/graphknows/services/graphknows')

from rich.console import Console
from rich.logging import RichHandler

console = Console(highlight=False)

logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s  %(message)s',
    datefmt='[%H:%M:%S]',
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=False, markup=False)],
    force=True,
)
for _ns in ('graphknows', 'graphgen', 'agent', 'search', 'extractor'):
    logging.getLogger(_ns).setLevel(logging.INFO)
for _ns in (
    'httpx', 'httpcore', 'urllib3', 'asyncio',
    'torch', 'transformers', 'sentence_transformers',
    'PIL', 'neo4j', 'filelock', 'datasets', 'tokenizers',
    'accelerate', 'hf_transfer', 'huggingface_hub',
):
    logging.getLogger(_ns).setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv('/root/graphknows/.env')

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.graph.agent import build_graph
from langchain_core.messages import HumanMessage

# ── Sessions config ────────────────────────────────────────────────────────────
SESSIONS = [
    {
        'session_id': 'ltm-test-s1-cbdc',
        'label': 'Session 1 — ECB Digital Euro / CBDC',
        'query': (
            'Investigate the ECB digital euro CBDC project: institutional actors, '
            'legislative timeline, political opposition from member states, '
            'and monetary sovereignty concerns. '
            'Write a concise EPRS-style intelligence brief for the ECON committee.'
        ),
    },
    {
        'session_id': 'ltm-test-s2-aiact',
        'label': 'Session 2 — EU AI Act Implementation',
        'query': (
            'Investigate the EU AI Act implementation: which institutions lead enforcement, '
            'which high-risk AI categories face the most pushback, '
            'and how does digital governance connect to EU financial regulation? '
            'Write a concise EPRS intelligence brief for the LIBE committee.'
        ),
    },
    {
        'session_id': 'ltm-test-s3-greendeal',
        'label': 'Session 3 — EU Green Deal Financing',
        'query': (
            'Investigate EU Green Deal financing mechanisms: the role of the EIB, '
            'InvestEU, carbon border adjustment, and member state fiscal capacity. '
            'How do ECB monetary policy and AI-driven ESG analytics shape investment flows? '
            'Write a concise EPRS intelligence brief for the ENVI committee.'
        ),
    },
]

OUTPUT_DIR = Path('/root/graphknows/output/ltm_test')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_TOOL_META: dict[str, tuple[str, str, str]] = {
    'search_web':      ('>', 'blue',    'Web Search'),
    'search_eu':       ('EU', 'yellow', 'EU Sources'),
    'search_archives': ('#', 'magenta', 'Archives'),
    'graphgen':        ('G', 'green',   'Graph Build'),
    'hybridrag':       ('R', 'cyan',    'RAG Retrieve'),
}

_PHASE_META: dict[str, tuple[str, str]] = {
    'scope':       ('bold yellow', 'SCOPE        Planning investigation strategy'),
    'recall':      ('bold cyan',   'RECALL       Querying long-term memory'),
    'investigate': ('bold blue',   'INVESTIGATE  Executing research plan'),
    'communicate': ('bold green',  'COMMUNICATE  Composing final brief'),
}


async def run_session(session_cfg: dict) -> dict:
    """Run one investigation session; return a trace dict.

    Mirrors run_investigation.py exactly:
      1. Create SessionGraphManager
      2. Register _persist_graph_to_sgm callback
      3. Stream events from build_graph
      4. Call merge_session_to_neo4j after the loop
      5. Sleep to let async LTM pipeline complete
    """
    sid = session_cfg['session_id']
    query = session_cfg['query']
    label = session_cfg['label']

    # ── Per-session mutable state (closed over by callbacks) ──────────────────
    _graph_nodes: list[dict] = []
    _graph_edges: list[dict] = []
    _phase_events: list[dict] = []

    console.print()
    console.print(Panel(
        Text(query, style='bold'),
        title=f'[bold cyan]{label}[/bold cyan]',
        subtitle=f'[dim]session={sid}[/dim]',
        border_style='cyan',
        padding=(1, 2),
    ))
    console.print()

    # ── Session graph manager ─────────────────────────────────────────────────
    from agent.session.manager import SessionGraphManager
    from agent.session.end import merge_session_to_neo4j as _merge

    _sgm = SessionGraphManager(
        memgraph_uri=os.environ.get('MEMGRAPH_URI', 'bolt://172.18.0.1:17687'),
        neo4j_uri=os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687'),
        neo4j_user=os.environ.get('NEO4J_USER', 'neo4j'),
        neo4j_password=os.environ['NEO4J_PASSWORD'],
    )
    _sgm.close_session(sid)  # purge any stale Memgraph data from prior runs

    async def _persist_graph_to_sgm(session_id: str, nodes: list, edges: list) -> None:
        """Write graph extraction results into SGM's in-process store."""
        def _sync() -> None:
            for node in nodes:
                props = dict(node.get('properties') or {})
                node_name = props.get('name') or node.get('id', '')
                _sgm.add_node(
                    session_id,
                    node_name,
                    node.get('label') or node.get('type') or 'Entity',
                    {'name': node_name, **props},
                )
            for edge in edges:
                _sgm.add_edge(
                    session_id,
                    str(edge.get('source', '')),
                    str(edge.get('target', '')),
                    edge.get('type', 'RELATED_TO'),
                    dict(edge.get('properties') or {}),
                )
        await asyncio.to_thread(_sync)
        mem = _sgm._mem_sessions.get(session_id, {})
        logging.getLogger('agent.session.persist').info(
            'persist_graph  session=%s  +%d nodes +%d edges  total=%d nodes %d edges',
            session_id, len(nodes), len(edges),
            len(mem.get('nodes', {})), len(mem.get('edges', [])),
        )

    async def broadcaster(session_id: str, payload: dict) -> None:
        ev_type = payload.get('type', '')
        ts = datetime.now(timezone.utc).isoformat()

        if ev_type == 'graph_update':
            gdata = payload.get('graph', {})
            if not payload.get('partial', True):
                _graph_nodes.clear()
                _graph_nodes.extend(gdata.get('nodes', []))
                _graph_edges.clear()
                _graph_edges.extend(gdata.get('edges', []))
            elif not _graph_nodes:
                _graph_nodes.extend(gdata.get('nodes', []))
                _graph_edges.extend(gdata.get('edges', []))

        elif ev_type == 'phase_change':
            phase = payload.get('phase', '')
            _phase_events.append({'ts': ts, 'phase': phase})
            style, phase_label = _PHASE_META.get(phase, ('bold white', phase.upper()))
            console.print()
            console.rule(f'[{style}]◉  {phase_label}[/{style}]', style=style)
            console.print()

    # ── Build and stream the agent graph ─────────────────────────────────────
    graph = build_graph(
        session_id=sid,
        openrouter_api_key=os.environ.get('OPENROUTER_API_KEY', ''),
        ws_broadcaster=broadcaster,
        persist_graph=_persist_graph_to_sgm,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()

    agent_invocations: list[dict] = []
    current_tokens: list[str] = []
    current_has_tool_calls: bool = False
    agent_turn: int = 0
    tool_calls: list[dict] = []
    active_starts: dict[str, dict] = {}
    workflow_nodes: list[str] = []

    async for event in graph.astream_events(
        {'messages': [HumanMessage(content=query)]},
        version='v2',
    ):
        ev_type: str = event.get('event', '')
        ev_name: str = event.get('name', '')
        ev_data: dict = event.get('data', {})
        ev_meta: dict = event.get('metadata', {})
        run_id: str = str(event.get('run_id', ''))
        lg_node: str = ev_meta.get('langgraph_node', '')
        ts_now = datetime.now(timezone.utc).isoformat()

        if lg_node and lg_node not in workflow_nodes and lg_node not in ('__start__', '__end__'):
            workflow_nodes.append(lg_node)

        if ev_type == 'on_chain_start' and ev_name == 'agent':
            agent_turn += 1
            current_tokens = []
            current_has_tool_calls = False
            if agent_turn == 2:
                await broadcaster(sid, {'type': 'phase_change', 'phase': 'investigate'})
            console.print(f'[dim]LLM turn {agent_turn}[/dim]')

        elif ev_type == 'on_chain_end' and ev_name == 'agent':
            msgs = (ev_data.get('output') or {}).get('messages', [])
            if msgs:
                last = msgs[-1]
                current_has_tool_calls = bool(getattr(last, 'tool_calls', None))
            agent_invocations.append({
                'tokens': current_tokens[:],
                'has_tool_calls': current_has_tool_calls,
            })
            if current_tokens:
                console.print()
            if not current_has_tool_calls:
                await broadcaster(sid, {'type': 'phase_change', 'phase': 'communicate'})

        elif ev_type == 'on_chat_model_stream' and lg_node == 'agent':
            chunk = ev_data.get('chunk')
            if chunk and hasattr(chunk, 'content') and chunk.content:
                current_tokens.append(chunk.content)
                console.print(chunk.content, end='', markup=False, highlight=False)

        elif ev_type == 'on_tool_start':
            icon, color, tlabel = _TOOL_META.get(ev_name, ('-', 'white', ev_name))
            active_starts[run_id] = {
                'name': ev_name,
                'inputs': ev_data.get('input', {}),
                'started_at': ts_now,
                'mono': time.monotonic(),
            }
            inp = ev_data.get('input', {})
            preview = inp.get('query', inp.get('session_id_arg', str(inp)[:60]))
            console.print(f'  [{color}]{icon} {tlabel}[/{color}]  [dim]{str(preview)[:80]}[/dim]')

        elif ev_type == 'on_tool_end':
            start_info = active_starts.pop(run_id, {})
            elapsed = time.monotonic() - start_info.get('mono', time.monotonic())
            raw = ev_data.get('output', '')
            output_str = str(raw.content) if hasattr(raw, 'content') else str(raw)
            tool_calls.append({
                'step': len(tool_calls) + 1,
                'tool': start_info.get('name', ev_name),
                'inputs': start_info.get('inputs', {}),
                'output': output_str[:2000],
                'duration_ms': int(elapsed * 1000),
                'started_at': start_info.get('started_at', ts_now),
                'finished_at': ts_now,
            })

    t_end = time.monotonic()
    finished_at = datetime.now(timezone.utc).isoformat()
    total_s = round(t_end - t_start, 2)

    # ── Session merge — Neo4j + LTM pipeline ─────────────────────────────────
    console.print()
    console.rule('[bold magenta]SESSION MERGE  Memgraph -> Neo4j LTM pipeline[/bold magenta]', style='magenta')
    merge_result: dict = {}
    try:
        merge_result = _merge(session_id=sid, sgm=_sgm, export_html=False)
        console.print(
            f'[bold magenta]merged {merge_result.get("merged_nodes", 0)} nodes, '
            f'{merge_result.get("merged_edges", 0)} edges -> Neo4j  '
            f'communities={merge_result.get("gds_communities", "?")}[/bold magenta]'
        )
    except Exception as exc:
        console.print(f'[red]Merge failed: {exc}[/red]')
        merge_result = {'error': str(exc)}

    # Allow async LTM tasks (WriteBackService: segment nodes + topic summarization) to finish
    console.print('[dim]Waiting 20s for async LTM pipeline (feature eng + topic summarization)...[/dim]')
    await asyncio.sleep(20)

    # ── Extract final answer ───────────────────────────────────────────────────
    answer_tokens: list[str] = []
    for inv in reversed(agent_invocations):
        if not inv['has_tool_calls']:
            answer_tokens = inv['tokens']
            break
    answer = ''.join(answer_tokens)

    trace = {
        'session_id': sid,
        'label': label,
        'query': query,
        'started_at': started_at,
        'finished_at': finished_at,
        'total_seconds': total_s,
        'tool_calls': tool_calls,
        'workflow_nodes_visited': workflow_nodes,
        'graph': {'nodes': _graph_nodes, 'edges': _graph_edges},
        'answer': answer,
        'phase_events': _phase_events,
        'session_merge': merge_result,
    }

    trace_path = OUTPUT_DIR / f'{sid}_trace.json'
    answer_path = OUTPUT_DIR / f'{sid}_answer.txt'
    trace_path.write_text(json.dumps(trace, default=str, indent=2))
    answer_path.write_text(answer)
    console.print(f'[dim]Trace -> {trace_path}[/dim]')

    return trace


# ── Verification helpers ───────────────────────────────────────────────────────

def _neo4j_check() -> dict:
    from neo4j import GraphDatabase
    uri = os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    pw = os.environ['NEO4J_PASSWORD']
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    results: dict = {}
    try:
        with driver.session() as s:
            results['entity_count'] = s.run("MATCH (e:Entity) RETURN count(e) as c").single()['c']
            results['entities_with_pagerank'] = s.run(
                "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL RETURN count(e) as c"
            ).single()['c']
            results['entities_with_community'] = s.run(
                "MATCH (e:Entity) WHERE e.community IS NOT NULL OR e.community_id IS NOT NULL "
                "RETURN count(e) as c"
            ).single()['c']
            results['segment_count'] = s.run("MATCH (s:Segment) RETURN count(s) as c").single()['c']
            results['topic_count'] = s.run("MATCH (t:Topic) RETURN count(t) as c").single()['c']
            results['appeared_in_edges'] = s.run(
                "MATCH ()-[r:APPEARED_IN]->() RETURN count(r) as c"
            ).single()['c']
            results['belongs_to_edges'] = s.run(
                "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as c"
            ).single()['c']
            results['top_entities'] = s.run(
                "MATCH (e:Entity) WHERE e.pagerank IS NOT NULL "
                "RETURN e.name as name, e.pagerank as pr, "
                "coalesce(e.community, e.community_id, -1) as community "
                "ORDER BY e.pagerank DESC LIMIT 8"
            ).data()
            results['topic_summaries'] = s.run(
                "MATCH (t:Topic) WHERE t.summary IS NOT NULL "
                "RETURN t.community_id as cid, t.summary as summary LIMIT 5"
            ).data()
            results['source_sessions'] = s.run(
                "MATCH (e:Entity) "
                "UNWIND coalesce(e.source_sessions, []) as sid "
                "RETURN DISTINCT sid ORDER BY sid"
            ).value()
    finally:
        driver.close()
    return results


async def _pgvector_check() -> dict:
    """Query pgvector ltm_topics via asyncpg."""
    import asyncpg

    pg_dsn = os.environ.get('PGVECTOR_DSN', '')
    if not pg_dsn:
        return {'error': 'PGVECTOR_DSN not set'}

    async def _run() -> dict:
        conn = await asyncpg.connect(pg_dsn, timeout=10)
        try:
            topic_count = await conn.fetchval("SELECT count(*) FROM ltm_topics")
            with_embedding = await conn.fetchval(
                "SELECT count(*) FROM ltm_topics WHERE embedding IS NOT NULL"
            )
            rows = await conn.fetch(
                "SELECT community_id, LEFT(summary, 80) as summary, entity_count "
                "FROM ltm_topics ORDER BY entity_count DESC NULLS LAST LIMIT 5"
            )
            return {
                'ltm_topics_count': topic_count or 0,
                'topics_with_embedding': with_embedding or 0,
                'sample_rows': [{'cid': r['community_id'], 'summary': r['summary'],
                                  'entity_count': r['entity_count']} for r in rows],
            }
        finally:
            await conn.close()

    try:
        return asyncio.get_event_loop().run_until_complete(_run())
    except Exception as e:
        return {'error': str(e)}


async def _recall_check(query: str) -> dict:
    """Direct ANN check over ltm_topics via pgvector."""
    try:
        import json as _json

        pg_dsn = os.environ.get('PGVECTOR_DSN', '')
        if not pg_dsn:
            return {'error': 'PGVECTOR_DSN not set'}

        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        emb = embedder.encode(query, normalize_embeddings=True).tolist()
        embedding_str = _json.dumps(emb)

        import asyncpg

        async def _run() -> list:
            conn = await asyncpg.connect(pg_dsn, timeout=5)
            try:
                rows = await conn.fetch(
                    "SELECT community_id, summary FROM ltm_topics "
                    "ORDER BY embedding <=> $1::vector LIMIT 3",
                    embedding_str,
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        rows = await _run()
        return {
            'query': query[:60],
            'results_count': len(rows),
            'top_results': [{'community_id': r['community_id'], 'summary': r['summary']} for r in rows],
        }
    except Exception as e:
        return {'error': str(e)}


def _cli_check() -> dict:
    import subprocess
    cli = '/root/graphknows/.venv/bin/graphknows-memory'
    env = {
        **os.environ,
        'PYTHONPATH': '/root/graphknows/services:/root/graphknows/services/graphknows',
    }
    results: dict = {}
    for cmd, args in [
        ('stats',        [cli, 'stats']),
        ('query_cbdc',   [cli, 'query', 'ECB digital euro monetary policy', '--top-k', '3']),
        ('query_ai',     [cli, 'query', 'EU AI Act enforcement digital governance', '--top-k', '3']),
        ('forget_help',  [cli, 'forget', '--help']),
    ]:
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=30,
                               cwd='/root/graphknows', env=env)
            results[cmd] = {'stdout': r.stdout.strip()[:600], 'returncode': r.returncode}
        except subprocess.TimeoutExpired:
            results[cmd] = {'error': 'timeout'}
        except Exception as e:
            results[cmd] = {'error': str(e)}
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    n_sessions = int(os.environ.get('LTM_TEST_SESSIONS', len(SESSIONS)))
    sessions_to_run = SESSIONS[:n_sessions]
    console.print()
    console.rule(f'[bold white]GraphKnows LTM Integration Test — {n_sessions} Session(s)[/bold white]')

    all_traces: list[dict] = []

    for i, session_cfg in enumerate(sessions_to_run):
        console.print()
        console.rule(f'[bold cyan]{session_cfg["label"]}[/bold cyan]', style='cyan')

        trace = await run_session(session_cfg)
        all_traces.append(trace)

        # ── Per-session snapshot ──────────────────────────────────────────────
        console.print()
        console.rule('[bold yellow]LTM State Snapshot[/bold yellow]', style='yellow')
        neo4j = _neo4j_check()
        pgv = await _pgvector_check()

        t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        t.add_column('metric', style='dim', no_wrap=True)
        t.add_column('value', style='bold')
        t.add_row('Session completed', session_cfg['label'])
        t.add_row('Duration', f'{trace["total_seconds"]}s')
        t.add_row('Tool calls', str(len(trace['tool_calls'])))
        t.add_row('Merged nodes', str(trace['session_merge'].get('merged_nodes', '?')))
        t.add_row('Merged edges', str(trace['session_merge'].get('merged_edges', '?')))
        t.add_row('GDS communities', str(trace['session_merge'].get('gds_communities', '?')))
        t.add_row('---', '---')
        t.add_row('LTM entities total', str(neo4j.get('entity_count', '?')))
        t.add_row('  with pagerank', str(neo4j.get('entities_with_pagerank', '?')))
        t.add_row('  with community', str(neo4j.get('entities_with_community', '?')))
        t.add_row('Segment nodes', str(neo4j.get('segment_count', '?')))
        t.add_row('Topic nodes (Neo4j)', str(neo4j.get('topic_count', '?')))
        t.add_row('APPEARED_IN edges', str(neo4j.get('appeared_in_edges', '?')))
        t.add_row('BELONGS_TO edges', str(neo4j.get('belongs_to_edges', '?')))
        t.add_row('Sessions in Neo4j', ', '.join(neo4j.get('source_sessions', [])) or '-')
        t.add_row('---', '---')
        t.add_row('ltm_topics (pgvector)', str(pgv.get('ltm_topics_count', '?')))
        t.add_row('  with embeddings', str(pgv.get('topics_with_embedding', '?')))
        console.print(t)

        if neo4j.get('top_entities'):
            console.print()
            console.print('[bold]Top entities by PageRank:[/bold]')
            for e in neo4j['top_entities'][:6]:
                console.print(
                    f"  {e['name'][:45]:45s}  pr={float(e.get('pr') or 0):.4f}  "
                    f"community={e.get('community', '?')}"
                )

        if neo4j.get('topic_summaries'):
            console.print()
            console.print('[bold]Topic summaries in Neo4j:[/bold]')
            for top in neo4j['topic_summaries']:
                console.print(f"  [cid={top['cid']}] {str(top['summary'])[:90]}")

        # From session 2 onward, test recall
        if i > 0:
            console.print()
            console.print('[bold]Recall ANN check:[/bold]')
            recall = await _recall_check(session_cfg['query'][:120])
            if 'error' in recall:
                console.print(f'  [red]FAIL: {recall["error"]}[/red]')
            else:
                console.print(f'  Returned {recall["results_count"]} topics')
                for r in recall.get('top_results', []):
                    console.print(f"  [cid={r['community_id']}] {str(r['summary'])[:80]}")

        if i < len(sessions_to_run) - 1:
            console.print()
            console.print('[dim]Pausing 5s before next session...[/dim]')
            await asyncio.sleep(5)

    # ── Final verification suite ───────────────────────────────────────────────
    console.print()
    console.rule('[bold green]FINAL LTM VERIFICATION SUITE[/bold green]', style='green')
    console.print()

    neo4j_final = _neo4j_check()
    pgv_final = await _pgvector_check()
    recall_final = await _recall_check('European Central Bank digital currency monetary sovereignty')
    cli_results = _cli_check()

    checks: list[tuple[str, bool]] = [
        ('Segment nodes in Neo4j >= 3',         neo4j_final.get('segment_count', 0) >= 3),
        ('Entities with pagerank > 0',           neo4j_final.get('entities_with_pagerank', 0) > 0),
        ('Entities with community_id > 0',       neo4j_final.get('entities_with_community', 0) > 0),
        ('Topic nodes in Neo4j > 0',             neo4j_final.get('topic_count', 0) > 0),
        ('APPEARED_IN edges > 0',                neo4j_final.get('appeared_in_edges', 0) > 0),
        ('ltm_topics in pgvector > 0',           pgv_final.get('ltm_topics_count', 0) > 0),
        ('ltm_topics with embeddings > 0',       pgv_final.get('topics_with_embedding', 0) > 0),
        ('Recall ANN returns results',           recall_final.get('results_count', 0) > 0),
        ('CLI stats exits 0',                    cli_results.get('stats', {}).get('returncode') == 0),
        ('CLI query exits 0',                    cli_results.get('query_cbdc', {}).get('returncode') == 0),
        ('CLI forget --help exits 0',            cli_results.get('forget_help', {}).get('returncode') == 0),
        ('All 3 sessions merged to Neo4j',
         len(neo4j_final.get('source_sessions', [])) >= 3),
    ]

    result_table = Table(box=box.ROUNDED, show_header=True, header_style='bold dim')
    result_table.add_column('Check', style='dim')
    result_table.add_column('Result', width=6, justify='center')

    all_pass = True
    for check_name, passed in checks:
        result_table.add_row(
            check_name,
            '[bold green]PASS[/bold green]' if passed else '[bold red]FAIL[/bold red]',
        )
        if not passed:
            all_pass = False

    console.print(result_table)

    console.print()
    console.print('[bold]CLI stats:[/bold]')
    console.print(cli_results.get('stats', {}).get('stdout', '(no output)'))

    console.print()
    console.print('[bold]CLI query — ECB digital euro monetary policy:[/bold]')
    console.print(cli_results.get('query_cbdc', {}).get('stdout', '(no output)')[:500])

    console.print()
    console.print('[bold]Recall — ECB digital currency monetary sovereignty:[/bold]')
    if 'error' in recall_final:
        console.print(f'[red]{recall_final["error"]}[/red]')
    else:
        for r in recall_final.get('top_results', []):
            console.print(f"  [cid={r['community_id']}] {str(r['summary'])[:100]}")

    # Save full report
    report = {
        'sessions': all_traces,
        'final_neo4j': neo4j_final,
        'final_pgvector': pgv_final,
        'checks': [{'check': c[0], 'passed': c[1]} for c in checks],
        'all_pass': all_pass,
        'cli_results': cli_results,
        'recall_final': recall_final,
    }
    report_path = OUTPUT_DIR / 'ltm_test_report.json'
    report_path.write_text(json.dumps(report, default=str, indent=2))

    console.print()
    if all_pass:
        console.rule('[bold green]ALL CHECKS PASSED[/bold green]', style='green')
    else:
        console.rule('[bold red]SOME CHECKS FAILED — see table above[/bold red]', style='red')

    console.print(f'[dim]Full report -> {report_path}[/dim]')
    console.print()


asyncio.run(main())
