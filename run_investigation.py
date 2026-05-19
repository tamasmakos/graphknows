"""Run a single investigation and emit a rich trace JSON for the HTML builder."""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, '/root/graphknows/services')
sys.path.insert(0, '/root/graphknows/services/agent')
sys.path.insert(0, '/root/graphknows/services/graphknows')

# ── Logging — must be configured before service imports touch their loggers ───
logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format='%(asctime)s.%(msecs)03d  %(name)-44s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    force=True,
)
for _ns in ('graphknows', 'graphgen', 'agent', 'search', 'extractor'):
    logging.getLogger(_ns).setLevel(logging.DEBUG)
# Silence noisy third-party namespaces
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.graph.agent import build_graph
from langchain_core.messages import HumanMessage

QUERY = (
    'Investigate the ECB digital euro CBDC project: institutional actors, '
    'legislative timeline, political opposition, and monetary sovereignty concerns. '
    'Write a full EPRS-style intelligence brief for the ECON committee.'
)
SESSION_ID = 'ecb-cbdc-001'
TRACE_PATH = '/root/graphknows/output/investigation_trace.json'
ANSWER_PATH = '/root/graphknows/output/agent_answer.txt'

console = Console(highlight=False)

# ── Tool display metadata ─────────────────────────────────────────────────────
_TOOL_META: dict[str, tuple[str, str, str]] = {
    'search_web':      ('🌐', 'blue',    'Web Search'),
    'search_eu':       ('🇪🇺', 'yellow', 'EU Sources'),
    'search_archives': ('📚', 'magenta', 'Archives'),
    'graphgen':        ('🧠', 'green',   'Graph Build'),
    'hybridrag':       ('🔍', 'cyan',    'RAG Retrieve'),
}

_PHASE_META: dict[str, tuple[str, str]] = {
    'scope':       ('bold yellow', 'SCOPE        Planning investigation strategy'),
    'investigate': ('bold blue',   'INVESTIGATE  Executing research plan'),
    'communicate': ('bold green',  'COMMUNICATE  Composing final brief'),
}


# ── Broadcaster state ─────────────────────────────────────────────────────────
_graph_nodes: list[dict] = []
_graph_edges: list[dict] = []
_lexical_nodes: list[dict] = []
_lexical_edges: list[dict] = []
_phase_events: list[dict] = []
_evidence_events: list[dict] = []


async def broadcaster(session_id: str, payload: dict) -> None:
    ev_type = payload.get('type', '')
    ts = datetime.now(timezone.utc).isoformat()

    if ev_type == 'graph_update':
        gdata = payload.get('graph', {})
        ldata = payload.get('lexical', {})
        is_partial = payload.get('partial', True)
        if not is_partial:
            _graph_nodes.clear()
            _graph_nodes.extend(gdata.get('nodes', []))
            _graph_edges.clear()
            _graph_edges.extend(gdata.get('edges', []))
            _lexical_nodes.clear()
            _lexical_nodes.extend(ldata.get('nodes', []))
            _lexical_edges.clear()
            _lexical_edges.extend(ldata.get('edges', []))
        elif not _graph_nodes:
            _graph_nodes.extend(gdata.get('nodes', []))
            _graph_edges.extend(gdata.get('edges', []))

    elif ev_type == 'phase_change':
        phase = payload.get('phase', '')
        _phase_events.append({'ts': ts, 'phase': phase})
        style, label = _PHASE_META.get(phase, ('bold white', phase.upper()))
        console.print()
        console.rule(f'[{style}]◉  {label}[/{style}]', style=style)
        console.print()

    elif ev_type == 'evidence_added':
        _evidence_events.append({
            'ts': ts,
            'title': payload.get('title', ''),
            'url': payload.get('url', ''),
            'source': payload.get('source', ''),
        })


def _tool_input_preview(inp: object) -> str:
    """Return a short human-readable summary of a tool's input."""
    if not isinstance(inp, dict):
        return str(inp)[:80]
    for key in ('query', 'session_id_arg', 'run_id'):
        if key in inp:
            return f'{key}={str(inp[key])[:80]}'
    return str(inp)[:80]


def _tool_output_preview(output_str: str, max_lines: int = 5) -> str:
    """Return the first few meaningful lines of a tool's output."""
    lines = [ln.strip() for ln in output_str.splitlines() if ln.strip()][:max_lines]
    return '\n'.join(lines)


async def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text(QUERY, style='bold'),
        title='[bold cyan]GraphKnows Investigator[/bold cyan]',
        subtitle=f'[dim]session={SESSION_ID}[/dim]',
        border_style='cyan',
        padding=(1, 2),
    ))
    console.print()

    # ── Session graph manager — created *before* build_graph so the
    # persist_graph callback can write GLiNER2 results into _mem_sessions ──────
    from agent.session.manager import SessionGraphManager
    from agent.session.end import merge_session_to_neo4j as _merge

    _sgm = SessionGraphManager(
        memgraph_uri=os.environ.get('MEMGRAPH_URI', 'bolt://172.18.0.1:17687'),
        neo4j_uri=os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687'),
        neo4j_user=os.environ.get('NEO4J_USER', 'neo4j'),
        neo4j_password=os.environ['NEO4J_PASSWORD'],
    )
    _persist_log = logging.getLogger('agent.session.persist')

    async def _persist_graph_to_sgm(sid: str, nodes: list, edges: list) -> None:
        """Persist GLiNER2/DSPy graph data into _sgm._mem_sessions for later merge."""
        if sid not in _sgm._mem_sessions:
            _sgm._mem_sessions[sid] = {'nodes': {}, 'edges': []}
        sess = _sgm._mem_sessions[sid]
        for n in nodes:
            sess['nodes'][n['id']] = n
        sess['edges'].extend(edges)
        _persist_log.info(
            'persist_graph  session=%s  +%d nodes +%d edges  total=%d nodes %d edges',
            sid, len(nodes), len(edges), len(sess['nodes']), len(sess['edges']),
        )

    graph = build_graph(
        session_id=SESSION_ID,
        groq_api_key=os.environ['GROQ_API_KEY'],
        ws_broadcaster=broadcaster,
        persist_graph=_persist_graph_to_sgm,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()

    # Per-agent-invocation tracking
    agent_invocations: list[dict] = []
    current_tokens: list[str] = []
    current_has_tool_calls: bool = False
    agent_turn: int = 0

    # Tool call tracking
    tool_calls: list[dict] = []
    active_starts: dict[str, dict] = {}  # run_id → {name, inputs, mono, ts}

    workflow_nodes: list[str] = []

    async for event in graph.astream_events(
        {'messages': [HumanMessage(content=QUERY)]},
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

        # ── Agent turn start ──────────────────────────────────────────────
        if ev_type == 'on_chain_start' and ev_name == 'agent':
            agent_turn += 1
            current_tokens = []
            current_has_tool_calls = False
            if agent_turn == 2:
                # First turn with tool results available → investigate phase
                await broadcaster(SESSION_ID, {'type': 'phase_change', 'phase': 'investigate'})
            console.print(
                f'[dim]┌─ LLM turn {agent_turn} ───────────────────────────────────────────[/dim]'
            )

        # ── Agent turn end ────────────────────────────────────────────────
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
                console.print()  # newline after inline token stream
            if current_has_tool_calls and msgs:
                tc_names = [tc.get('name', '?') for tc in (msgs[-1].tool_calls or [])]
                console.print(f'[dim]└─ dispatching: {", ".join(tc_names)}[/dim]')
            else:
                # No tool calls → this is the final report (communicate phase)
                await broadcaster(SESSION_ID, {'type': 'phase_change', 'phase': 'communicate'})
                console.print(f'[dim]└─ turn {agent_turn} complete[/dim]')
            console.print()

        # ── LLM token streaming ───────────────────────────────────────────
        elif ev_type == 'on_chat_model_stream' and lg_node == 'agent':
            chunk = ev_data.get('chunk')
            if chunk and hasattr(chunk, 'content') and chunk.content:
                current_tokens.append(chunk.content)
                # markup=False prevents rich from misinterpreting [ in LLM output
                console.print(chunk.content, end='', markup=False, highlight=False)

        # ── Tool start ────────────────────────────────────────────────────
        elif ev_type == 'on_tool_start':
            icon, color, label = _TOOL_META.get(ev_name, ('🔧', 'white', ev_name))
            active_starts[run_id] = {
                'name': ev_name,
                'inputs': ev_data.get('input', {}),
                'started_at': ts_now,
                'mono': time.monotonic(),
            }
            preview = _tool_input_preview(ev_data.get('input', {}))
            console.print(
                f'  [{color}]{icon} {label}[/{color}]'
                f'  [dim italic]{preview}[/dim italic]',
            )

        # ── Tool end ──────────────────────────────────────────────────────
        elif ev_type == 'on_tool_end':
            start_info = active_starts.pop(run_id, {})
            elapsed = time.monotonic() - start_info.get('mono', time.monotonic())
            duration_ms = int(elapsed * 1000)

            raw_output = ev_data.get('output', '')
            output_str = (
                str(raw_output.content) if hasattr(raw_output, 'content') else str(raw_output)
            )

            doc_count = len([
                ln for ln in output_str.splitlines()
                if ln and ln[0].isdigit() and '. ' in ln
            ])

            tool_calls.append({
                'step': len(tool_calls) + 1,
                'tool': start_info.get('name', ev_name),
                'inputs': start_info.get('inputs', ev_data.get('input', {})),
                'output': output_str[:3000],
                'doc_count': doc_count,
                'started_at': start_info.get('started_at', ts_now),
                'finished_at': ts_now,
                'duration_ms': duration_ms,
            })

            icon, color, label = _TOOL_META.get(ev_name, ('🔧', 'white', ev_name))
            preview = _tool_output_preview(output_str)
            subtitle_parts = [f'{duration_ms} ms']
            if doc_count:
                subtitle_parts.append(f'{doc_count} docs')

            console.print(Panel(
                Text(preview[:500] if preview else output_str[:300], overflow='fold'),
                title=(
                    f'[{color}]{icon} {label}[/{color}]'
                    f'  [dim]{" · ".join(subtitle_parts)}[/dim]'
                ),
                border_style=color,
                padding=(0, 1),
            ))
            console.print()

    t_end = time.monotonic()
    finished_at = datetime.now(timezone.utc).isoformat()
    total_seconds = round(t_end - t_start, 2)

    # ── Session merge: Memgraph → Neo4j ──────────────────────────────────────
    console.print()
    console.rule('[bold magenta]◎  SESSION MERGE  Memgraph → Neo4j[/bold magenta]', style='magenta')
    console.print()
    merge_result: dict = {}
    try:
        merge_result = _merge(
            session_id=SESSION_ID,
            sgm=_sgm,
            export_html=os.environ.get('EXPORT_GRAPH_HTML', '').lower() == 'true',
        )
        merged_nodes = merge_result.get('merged_nodes', 0)
        merged_edges = merge_result.get('merged_edges', 0)
        console.print(
            f'[bold magenta]✓  merged {merged_nodes} nodes, {merged_edges} edges → Neo4j[/bold magenta]'
        )
        if 'graph_html_path' in merge_result:
            console.print(f'[dim]   Graph HTML → {merge_result["graph_html_path"]}[/dim]')
        if 'error' in merge_result:
            console.print(f'[red]   Merge error: {merge_result["error"]}[/red]')
    except Exception as _merge_exc:
        console.print(f'[red]Session merge failed: {_merge_exc}[/red]')
        merge_result = {'error': str(_merge_exc)}
    console.print()

    # ── Derive final answer ───────────────────────────────────────────────────
    answer_tokens: list[str] = []
    for inv in reversed(agent_invocations):
        if not inv['has_tool_calls']:
            answer_tokens = inv['tokens']
            break
    answer = ''.join(answer_tokens)

    # ── Persist outputs ───────────────────────────────────────────────────────
    with open(ANSWER_PATH, 'w') as fh:
        fh.write(answer)

    trace = {
        'query': QUERY,
        'session_id': SESSION_ID,
        'started_at': started_at,
        'finished_at': finished_at,
        'total_seconds': total_seconds,
        'tool_calls': tool_calls,
        'workflow_nodes_visited': workflow_nodes,
        'graph': {'nodes': _graph_nodes, 'edges': _graph_edges},
        'lexical': {'nodes': _lexical_nodes, 'edges': _lexical_edges},
        'answer': answer,
        'evidence_added': _evidence_events,
        'phase_events': _phase_events,
        'session_merge': merge_result,
    }
    with open(TRACE_PATH, 'w') as fh:
        json.dump(trace, fh, default=str, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule('[bold green]Investigation complete[/bold green]', style='green')
    console.print()

    stats = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    stats.add_column('metric', style='dim', no_wrap=True)
    stats.add_column('value', style='bold')
    stats.add_row('Duration',       f'{total_seconds} s')
    stats.add_row('Tool calls',     str(len(tool_calls)))
    stats.add_row('LLM turns',      str(agent_turn))
    stats.add_row('Evidence items', str(len(_evidence_events)))
    stats.add_row('Graph nodes',    str(len(_graph_nodes)))
    stats.add_row('Graph edges',    str(len(_graph_edges)))
    stats.add_row('Neo4j merged',   f'{merge_result.get("merged_nodes", "—")} nodes  {merge_result.get("merged_edges", "—")} edges')
    stats.add_row('Answer length',  f'{len(answer):,} chars')
    phases_str = ' → '.join(p['phase'] for p in _phase_events) or '—'
    stats.add_row('Phases',         phases_str)
    console.print(stats)

    # Tool breakdown table
    if tool_calls:
        console.print()
        breakdown = Table(box=box.SIMPLE, show_header=True, header_style='bold dim')
        breakdown.add_column('#',    width=3,  style='dim')
        breakdown.add_column('Tool', width=16)
        breakdown.add_column('ms',   width=6,  justify='right')
        breakdown.add_column('Docs', width=5,  justify='right')
        breakdown.add_column('Input preview', style='dim')
        for tc in tool_calls:
            icon, color, label = _TOOL_META.get(tc['tool'], ('🔧', 'white', tc['tool']))
            inp_preview = _tool_input_preview(tc['inputs'])[:55]
            breakdown.add_row(
                str(tc['step']),
                f'[{color}]{icon} {label}[/{color}]',
                str(tc['duration_ms']),
                str(tc['doc_count']) if tc['doc_count'] else '—',
                inp_preview,
            )
        console.print(breakdown)

    # Full answer panel
    if answer:
        console.print()
        console.print(Panel(
            Text(answer),
            title='[bold green]Intelligence Brief[/bold green]',
            subtitle=f'[dim]{len(answer):,} chars · saved → {ANSWER_PATH}[/dim]',
            border_style='green',
            padding=(1, 2),
        ))

    console.print()
    console.print(f'[dim]Trace → {TRACE_PATH}[/dim]')
    console.print()


asyncio.run(main())
