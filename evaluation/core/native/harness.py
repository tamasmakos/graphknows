"""Native MCP-level evaluation harness for graphknows."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from evaluation.core.reporting.metrics import ExpectedEvidence, ToolObservation


class MCPToolClient(Protocol):
    """Protocol implemented by real and fake MCP clients."""

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class NativeConfig:
    """Configuration for native graphknows MCP evaluation."""

    command: str = "graphknows-mcp"
    initialize: bool = True
    request_timeout_s: float = 120.0
    output_dir: Path = Path("evaluation/results/native")
    top_k: int = 5
    scope: str = "both"
    cleanup: bool = True


@dataclass
class NativeEvaluationRun:
    """State and observations for one native evaluation run."""

    run_id: str
    config: NativeConfig
    observations: list[ToolObservation] = field(default_factory=list)
    eval_sessions: set[str] = field(default_factory=set)

    def session_id(self, case: str) -> str:
        session_id = f"{self.run_id}-{case}"
        self.eval_sessions.add(session_id)
        return session_id


async def measured_call(
    client: MCPToolClient,
    run: NativeEvaluationRun,
    case_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    expected: ExpectedEvidence | None = None,
    cold_start: bool = False,
) -> ToolObservation:
    start = time.perf_counter()
    try:
        result = await client.call_tool(tool_name, arguments or {})
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        observation = ToolObservation(
            case_id=case_id,
            tool_name=tool_name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            expected=expected,
            cold_start=cold_start,
        )
    else:
        latency_ms = (time.perf_counter() - start) * 1000
        observation = ToolObservation(
            case_id=case_id,
            tool_name=tool_name,
            success=True,
            latency_ms=latency_ms,
            result=result,
            expected=expected,
            cold_start=cold_start,
        )
    run.observations.append(observation)
    return observation


async def cleanup_eval_sessions(client: MCPToolClient, run: NativeEvaluationRun) -> None:
    """Purge only scoped eval sessions created by this run."""
    for session_id in sorted(run.eval_sessions):
        if not session_id.startswith("eval-"):
            raise ValueError(f"refusing to purge non-eval session: {session_id}")
        await measured_call(
            client,
            run,
            case_id=f"cleanup-{session_id}",
            tool_name="memory_purge",
            arguments={"scope": "both", "session_id": session_id},
        )


async def scenario_health(client: MCPToolClient, run: NativeEvaluationRun) -> None:
    await measured_call(client, run, "health", "memory_doctor", cold_start=True)
    await measured_call(client, run, "health", "memory_stats", {"session_id": ""})
    await measured_call(client, run, "taxonomy", "list_taxonomies")


async def scenario_stm_recall(client: MCPToolClient, run: NativeEvaluationRun) -> None:
    session_id = run.session_id("stm")
    text = (
        "Graphknows evaluation seed: Project Aurora stores retry budgets in pgvector, "
        "promotes durable entities into Neo4j, and keeps live agent turns in Memgraph. "
        "The responsible engineer is Ada Lovelace and the release codename is Helios."
    )
    await measured_call(
        client,
        run,
        "stm-ingest",
        "memory_ingest",
        {"text": text, "session_id": session_id, "title": "native-stm-seed"},
    )
    await measured_call(
        client,
        run,
        "stm-recall",
        "memory_recall",
        {
            "query": "Which release codename is tied to Project Aurora?",
            "session_id": session_id,
            "top_k": run.config.top_k,
            "scope": "stm",
        },
        expected=ExpectedEvidence(terms=("helios", "project aurora"), entities=("ada lovelace",)),
    )


async def scenario_ltm_recall(client: MCPToolClient, run: NativeEvaluationRun) -> None:
    session_id = run.session_id("ltm")
    text = (
        "The Long Memory Council approved the Helios migration. "
        "Memgraph tracks short term graph state, pgvector stores semantic chunks, "
        "and Neo4j stores long term entity communities."
    )
    await measured_call(
        client,
        run,
        "ltm-ingest",
        "memory_ingest",
        {"text": text, "session_id": session_id, "title": "native-ltm-seed"},
    )
    await measured_call(client, run, "ltm-flush", "memory_flush", {"session_id": session_id})
    await measured_call(client, run, "ltm-flush-idempotent", "memory_flush", {"session_id": session_id})
    expected = ExpectedEvidence(terms=("helios", "neo4j"), entities=("long memory council",))
    await measured_call(
        client,
        run,
        "ltm-recall",
        "memory_recall",
        {
            "query": "Where are long term entity communities stored?",
            "session_id": session_id,
            "top_k": run.config.top_k,
            "scope": "ltm",
        },
        expected=expected,
    )
    await measured_call(
        client,
        run,
        "both-recall",
        "memory_recall",
        {
            "query": "How does Helios split state across Memgraph pgvector and Neo4j?",
            "session_id": session_id,
            "top_k": run.config.top_k,
            "scope": run.config.scope,
        },
        expected=expected,
    )
    await measured_call(client, run, "ltm-query", "ltm_query", {"query": "Helios", "top_k": 5})
    await measured_call(client, run, "ltm-entity", "ltm_entity", {"name": "Neo4j"})
    await measured_call(client, run, "ltm-topics", "ltm_topics", {"min_entities": 1})
    await measured_call(client, run, "recall-document", "recall_by_document", {"query": "Helios Neo4j", "top_k": 5})
    await measured_call(
        client,
        run,
        "recall-dialog",
        "recall_by_dialog",
        {"query": "Helios Neo4j", "top_k": 5, "session_filter": session_id},
    )
    await measured_call(client, run, "recall-topic", "recall_by_topic", {"query": "Helios Neo4j", "top_k": 5})


async def scenario_corpus(client: MCPToolClient, run: NativeEvaluationRun) -> None:
    session_id = run.session_id("corpus")
    documents = [
        {
            "id": "native-corpus-1",
            "title": "Native Corpus Seed",
            "source": "evaluation/native",
            "text": (
                "Atlas Memory Review says Project Helios uses corpus_ingest "
                "to promote document collections into long term memory."
            ),
        }
    ]
    await measured_call(
        client,
        run,
        "corpus-ingest",
        "corpus_ingest",
        {"documents": documents, "session_id": session_id},
    )


SCENARIOS: tuple[Callable[[MCPToolClient, NativeEvaluationRun], Awaitable[None]], ...] = (
    scenario_health,
    scenario_stm_recall,
    scenario_ltm_recall,
    scenario_corpus,
)


async def run_native_evaluation(
    client: MCPToolClient,
    config: NativeConfig | None = None,
    run_id: str | None = None,
) -> NativeEvaluationRun:
    cfg = config or NativeConfig()
    rid = run_id or f"eval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run = NativeEvaluationRun(run_id=rid, config=cfg)
    try:
        for scenario in SCENARIOS:
            await scenario(client, run)
    finally:
        if cfg.cleanup:
            await cleanup_eval_sessions(client, run)
    return run
