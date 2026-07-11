"""The shared benchmark pipeline.

One workflow, run per ingest unit: ingest the unit's documents once, then for
each question retrieve → generate → LLM-judge. Each benchmark plugs in via a
:class:`BenchmarkAdapter` that owns its prompts and judge; everything else —
MCP lifecycle, ingestion, retrieval, diagnostics, checkpointing, purging — is
identical across benchmarks and lives here.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import (
    CaseResult,
    EvalCase,
    IngestResult,
    IngestUnit,
    RetrievedPassage,
    RunReport,
)
from evaluation.common.ingestion import Ingestor
from evaluation.common.lexical import gold_context_coverage, gold_max_overlap, gold_reachable
from evaluation.common.llm import ChatLLM, CreditExhaustedError, resolve_model
from evaluation.common.reporting import summarize, write_report
from evaluation.common.retrieval import Retriever
from graphknows.client import GraphKnowsMCPClient

log = logging.getLogger(__name__)


class BenchmarkAdapter(abc.ABC):
    """Benchmark-specific behavior: prompts, answer post-processing, judging."""

    name: str = "benchmark"

    def __init__(self, config: BaseEvalConfig, answerer: ChatLLM, judge: ChatLLM) -> None:
        self.config = config
        self.answerer = answerer
        self.judge = judge

    @abc.abstractmethod
    async def answer(
        self, case: EvalCase, unit: IngestUnit, passages: list[RetrievedPassage]
    ) -> tuple[str, dict[str, Any]]:
        """Generate an answer; return ``(answer, usage_metrics)``."""

    @abc.abstractmethod
    async def judge_case(self, case: EvalCase, answer: str) -> tuple[float, dict[str, Any]]:
        """Score an answer; return ``(score_in_0_1, judge_metrics)``."""

    def lexical_metrics(self, case: EvalCase, answer: str) -> dict[str, Any]:
        """Optional cheap per-benchmark diagnostics (no LLM calls)."""
        return {}


def build_llms(config: BaseEvalConfig) -> tuple[ChatLLM, ChatLLM]:
    """Construct the (answerer, judge) pair from config + environment."""
    gen_model = resolve_model(config.generation_model, "OPENROUTER_MODEL", "LLM_MODEL")
    judge_model = resolve_model(config.judge_model, "JUDGE_MODEL") or gen_model
    answerer = ChatLLM(gen_model, max_tokens=4096)
    judge = ChatLLM(judge_model, max_tokens=2048)
    return answerer, judge


def _is_credit_exhaustion(exc: Exception) -> bool:
    """True when *exc* is an OpenRouter out-of-credits error (vs a real bug)."""
    if isinstance(exc, CreditExhaustedError):
        return True
    msg = str(exc).lower()
    return "credit" in msg or "402" in msg or "payment" in msg


def _avg_score(passages: list[RetrievedPassage]) -> float:
    scores = [p.score for p in passages if isinstance(p.score, (int, float))]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _checkpoint(
    run_id: str, adapter: BenchmarkAdapter, config: BaseEvalConfig, cases: list[CaseResult]
) -> None:
    """Persist results so far; a later crash never loses completed units."""
    try:
        write_report(
            RunReport(
                run_id=run_id,
                benchmark=adapter.name,
                config=config.model_dump(mode="json"),
                cases=list(cases),
                summary=summarize(cases),
            ),
            config.output_dir,
        )
    except Exception as exc:
        log.warning("Checkpoint write failed (non-fatal): %s", exc)


def _evaluation_env() -> dict[str, str]:
    """Environment overrides handed to the MCP subprocess."""
    env: dict[str, str] = {}
    if os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE"):
        env["GRAPHKNOWS_ONTOLOGY_FILE"] = os.environ["GRAPHKNOWS_ONTOLOGY_FILE"]
    return env


def build_case_result(
    case: EvalCase,
    *,
    config: BaseEvalConfig,
    session_id: str,
    passages: list[RetrievedPassage],
    probe: list[RetrievedPassage],
    context_passages: list[RetrievedPassage],
    generated_answer: str,
    score: float,
    judge_metrics: dict[str, Any],
    lexical: dict[str, Any],
    gen_usage: dict[str, Any],
    ingest: IngestResult,
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        gold=case.gold,
        category=case.category,
        group_id=case.group_id,
        passages=passages,
        generated_answer=generated_answer,
        score=score,
        metrics={
            "judge_score": score,
            **judge_metrics,
            **lexical,
            "gold_max_token_overlap": round(gold_max_overlap(case.gold, probe), 3),
            "gold_reachable_at_probe": gold_reachable(case.gold, probe),
            # Fraction of gold tokens present in the passages the GENERATOR saw:
            # high + wrong answer = generation failure, low = retrieval miss.
            "gold_context_coverage": round(gold_context_coverage(case.gold, context_passages), 3),
            "avg_hit_score": _avg_score(passages),
            "recall_hit_count": len(passages),
            "ltm_graph_hit_count": sum(1 for p in passages if p.is_ltm_graph),
            "ingested_chunks": ingest.chunks,
            "ingested_entities": ingest.entities,
            "ingest_elapsed_s": ingest.ingest_elapsed_s,
            "ltm_entity_count": ingest.ltm_entity_count,
            "flush_nodes": ingest.flush_nodes,
            "flush_edges": ingest.flush_edges,
            "flush_elapsed_s": ingest.flush_elapsed_s,
            "ingest_prompt_tokens": ingest.ingest_prompt_tokens,
            "ingest_completion_tokens": ingest.ingest_completion_tokens,
            "ingest_cost_usd": ingest.ingest_cost_usd,
            **gen_usage,
        },
        metadata={
            "session_id": session_id,
            "recall_scope": config.recall_scope,
            "promote_to_ltm": config.promote_to_ltm,
            "top_k": config.top_k,
        },
    )


async def _run_unit(
    unit: IngestUnit,
    *,
    adapter: BenchmarkAdapter,
    config: BaseEvalConfig,
    run_id: str,
    ingestor: Ingestor,
    retriever: Retriever,
    client: GraphKnowsMCPClient,
    progress: tuple[int, int],
    on_progress,
) -> list[CaseResult]:
    """Ingest one unit, then retrieve+generate+judge each of its questions."""
    case_idx, total = progress
    session_id = f"{config.session_prefix}-{run_id}-{unit.group_id}"
    results: list[CaseResult] = []

    print(f"\n[{unit.group_id}] ingesting {len(unit.documents)} docs …", flush=True)
    try:
        ingest = await ingestor.ingest(session_id, unit.documents)
        print(
            f"[{unit.group_id}] ingest done: chunks={ingest.chunks} "
            f"entities={ingest.entities} flush_nodes={ingest.flush_nodes}",
            flush=True,
        )

        for case in unit.cases:
            probe = await retriever.query(session_id, case.question, k=config.probe_k)
            passages = probe[: config.top_k]
            context_passages = probe[: config.gen_context_k]

            answer, gen_usage = await adapter.answer(case, unit, context_passages)
            score, judge_metrics = await adapter.judge_case(case, answer)
            lexical = adapter.lexical_metrics(case, answer)

            result = build_case_result(
                case,
                config=config,
                session_id=session_id,
                passages=passages,
                probe=probe,
                context_passages=context_passages,
                generated_answer=answer,
                score=score,
                judge_metrics=judge_metrics,
                lexical=lexical,
                gen_usage=gen_usage,
                ingest=ingest,
            )
            results.append(result)

            case_idx += 1
            preview = answer[:50].replace("\n", " ")
            print(
                f"  [{case_idx:3d}/{total}] {case.case_id}  "
                f"judge={score:.2f}  gen={preview!r}",
                flush=True,
            )
            if on_progress is not None and len(results) % 30 == 0:
                on_progress(results)
    finally:
        if config.purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return results


async def run_units(
    adapter: BenchmarkAdapter,
    config: BaseEvalConfig,
    units: list[IngestUnit],
    *,
    run_id: str,
) -> RunReport:
    """Run all units through the pipeline and return a populated report."""
    total = sum(len(u.cases) for u in units)
    cases: list[CaseResult] = []

    async with GraphKnowsMCPClient(
        config.command,
        request_timeout_s=config.request_timeout_s,
        env=_evaluation_env(),
    ) as client:
        if config.pre_run_purge_ltm:
            log.info("Purging all LTM before the run (pre_run_purge_ltm=True)")
            await client.call_tool("memory_purge", {"scope": "ltm", "session_id": ""})

        ingestor = Ingestor(client, config)
        retriever = Retriever(client, config)

        for i, unit in enumerate(units, 1):

            def _progress_ckpt(partial: list[CaseResult], _done: list[CaseResult] = cases) -> None:
                _checkpoint(run_id, adapter, config, _done + partial)

            try:
                results = await _run_unit(
                    unit,
                    adapter=adapter,
                    config=config,
                    run_id=run_id,
                    ingestor=ingestor,
                    retriever=retriever,
                    client=client,
                    progress=(len(cases), total),
                    on_progress=_progress_ckpt,
                )
            except Exception as exc:
                if _is_credit_exhaustion(exc):
                    # No traceback: out-of-credits is an operational stop, not a bug.
                    log.error(  # noqa: TRY400
                        "Stopping after %d/%d units — OpenRouter credits exhausted: %s",
                        i - 1, len(units), exc,
                    )
                else:
                    log.exception(
                        "Stopping after %d/%d units — unit %s failed",
                        i - 1, len(units), unit.group_id,
                    )
                break
            cases.extend(results)
            _checkpoint(run_id, adapter, config, cases)
            log.info(
                "Checkpoint: %d cases after unit %s (%d/%d)",
                len(cases), unit.group_id, i, len(units),
            )

    return RunReport(
        run_id=run_id,
        benchmark=adapter.name,
        config=config.model_dump(mode="json"),
        cases=cases,
        summary=summarize(cases),
    )
