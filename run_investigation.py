"""Run a single investigation and emit a rich trace JSON for the HTML builder."""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, '/root/graphknows/services')
sys.path.insert(0, '/root/graphknows/services/agent')
sys.path.insert(0, '/root/graphknows/services/graphknows')

from dotenv import load_dotenv
load_dotenv('/root/graphknows/.env')

from agent.langgraph_agent import build_graph
from langchain_core.messages import HumanMessage

QUERY = (
    'Investigate the ECB digital euro CBDC project: institutional actors, '
    'legislative timeline, political opposition, and monetary sovereignty concerns. '
    'Write a full EPRS-style intelligence brief for the ECON committee.'
)
SESSION_ID = 'ecb-cbdc-001'
TRACE_PATH = '/root/graphknows/output/investigation_trace.json'
ANSWER_PATH = '/root/graphknows/output/agent_answer.txt'

# ── Broadcaster state ──────────────────────────────────────────────────────────
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
            # Accept the first partial so we always have something
            _graph_nodes.extend(gdata.get('nodes', []))
            _graph_edges.extend(gdata.get('edges', []))
    elif ev_type == 'phase_change':
        _phase_events.append({'ts': ts, 'phase': payload.get('phase', '')})
    elif ev_type == 'evidence_added':
        _evidence_events.append({
            'ts': ts,
            'title': payload.get('title', ''),
            'url': payload.get('url', ''),
            'source': payload.get('source', ''),
        })


async def main() -> None:
    graph = build_graph(
        session_id=SESSION_ID,
        groq_api_key=os.environ['GROQ_API_KEY'],
        ws_broadcaster=broadcaster,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.monotonic()

    # Per-agent-invocation token buffers
    agent_invocations: list[dict] = []
    current_tokens: list[str] = []
    current_has_tool_calls: bool = False

    # Tool call tracking
    tool_calls: list[dict] = []
    active_starts: dict[str, dict] = {}  # run_id → {name, inputs, mono, ts}

    # LangGraph workflow nodes in visitation order
    workflow_nodes: list[str] = []

    print(f'Starting investigation: {QUERY[:80]}...')
    print()

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

        # Track LangGraph workflow node order
        if lg_node and lg_node not in workflow_nodes and lg_node not in ('__start__', '__end__'):
            workflow_nodes.append(lg_node)

        # ── Agent node boundaries ─────────────────────────────────────────
        if ev_type == 'on_chain_start' and ev_name == 'agent':
            current_tokens = []
            current_has_tool_calls = False

        elif ev_type == 'on_chain_end' and ev_name == 'agent':
            msgs = (ev_data.get('output') or {}).get('messages', [])
            if msgs:
                last = msgs[-1]
                current_has_tool_calls = bool(getattr(last, 'tool_calls', None))
            agent_invocations.append({
                'tokens': current_tokens[:],
                'has_tool_calls': current_has_tool_calls,
            })

        # ── LLM token streaming ───────────────────────────────────────────
        elif ev_type == 'on_chat_model_stream' and lg_node == 'agent':
            chunk = ev_data.get('chunk')
            if chunk and hasattr(chunk, 'content') and chunk.content:
                current_tokens.append(chunk.content)

        # ── Tool lifecycle ────────────────────────────────────────────────
        elif ev_type == 'on_tool_start':
            active_starts[run_id] = {
                'name': ev_name,
                'inputs': ev_data.get('input', {}),
                'started_at': ts_now,
                'mono': time.monotonic(),
            }
            inp_preview = str(ev_data.get('input', {}))[:120]
            print(f'  [TOOL START] {ev_name}  inputs={inp_preview}')

        elif ev_type == 'on_tool_end':
            start_info = active_starts.pop(run_id, {})
            elapsed = time.monotonic() - start_info.get('mono', time.monotonic())
            duration_ms = int(elapsed * 1000)

            raw_output = ev_data.get('output', '')
            if hasattr(raw_output, 'content'):
                output_str = str(raw_output.content)
            else:
                output_str = str(raw_output)

            # Count result documents (heuristic: numbered lines like "1. Title")
            doc_count = len([ln for ln in output_str.splitlines() if ln and ln[0].isdigit() and '. ' in ln])

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
            print(f'  [TOOL END]   {ev_name}  ({duration_ms}ms)  docs≈{doc_count}  '
                  f'output={output_str[:80].strip()!r}…')

    t_end = time.monotonic()
    finished_at = datetime.now(timezone.utc).isoformat()
    total_seconds = round(t_end - t_start, 2)

    # Derive final answer: tokens from the last non-tool-call agent invocation
    answer_tokens: list[str] = []
    for inv in reversed(agent_invocations):
        if not inv['has_tool_calls']:
            answer_tokens = inv['tokens']
            break
    answer = ''.join(answer_tokens)

    # ── Persist outputs ────────────────────────────────────────────────────
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
    }

    with open(TRACE_PATH, 'w') as fh:
        json.dump(trace, fh, default=str, indent=2)

    print()
    print(f'=== DONE in {total_seconds}s ===')
    print(f'Tool calls: {len(tool_calls)}')
    print(f'Answer length: {len(answer)} chars')
    print(f'Graph: {len(_graph_nodes)} nodes, {len(_graph_edges)} edges')
    print(f'Trace saved → {TRACE_PATH}')
    print()
    print('=== ANSWER (first 600 chars) ===')
    print(answer[:600])


asyncio.run(main())
