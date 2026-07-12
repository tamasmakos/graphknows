"""The shared benchmark pipeline.

One workflow, run per ingest unit: ingest the unit's documents once, then for
each question retrieve → generate → LLM-judge. Each benchmark plugs in via a
:class:`BenchmarkAdapter` that owns its prompts and judge; everything else —
MCP lifecycle, ingestion, retrieval, diagnostics, checkpointing, purging — is
identical across benchmarks and lives here.
"""

from __future__ import annotations

import abc
import asyncio
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
from evaluation.common.dspy_judge import DSPyJudge
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

    def __init__(self, config: BaseEvalConfig, answerer: ChatLLM, judge: DSPyJudge) -> None:
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


def build_llms(config: BaseEvalConfig) -> tuple[ChatLLM, DSPyJudge]:
    """Construct the (httpx answerer, DSPy judge) pair from config + environment."""
    gen_model = resolve_model(config.generation_model, "OPENROUTER_MODEL", "LLM_MODEL")
    judge_model = resolve_model(config.judge_model, "JUDGE_MODEL") or gen_model
    answerer = ChatLLM(gen_model, max_tokens=4096)
    judge = DSPyJudge(judge_model)
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


def unit_namespace(config: BaseEvalConfig, group_id: str) -> str:
    """Physical namespace for one ingest unit: ``<benchmark>_<mode>_<group_id>``.

    Every conversation gets its OWN database pair, so conversations share no graph
    (entities/topics/PageRank/communities never merge across them) — the paper's
    "memory built per conversation", and the property that makes concurrent
    ingest safe. The server sanitizes the name; pass the same raw string to
    ingest / query / drop so it maps consistently.
    """
    return f"{config.namespace}_{group_id}"


class _Progress:
    """Shared, lock-guarded case counter for live progress across concurrent units."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._done = 0
        self._lock = asyncio.Lock()

    async def bump(self) -> int:
        async with self._lock:
            self._done += 1
            return self._done


async def _run_unit(
    unit: IngestUnit,
    *,
    adapter: BenchmarkAdapter,
    config: BaseEvalConfig,
    ingestor: Ingestor,
    retriever: Retriever,
    client: GraphKnowsMCPClient,
    counter: _Progress,
    reuse_ingest: bool = False,
) -> list[CaseResult]:
    """Ingest one unit into its own namespace, then answer its questions.

    Ingest is sequential (one unit's docs share its graph); questions run
    concurrently (bounded by ``question_concurrency``) since retrieve→generate→
    judge is read-only against the DB with no shared writes.
    """
    # Deterministic per (benchmark, unit) so a --no-ingest re-run finds the same
    # session in the namespace instead of re-ingesting.
    session_id = f"{config.session_prefix}-{unit.group_id}"
    namespace = unit_namespace(config, unit.group_id)

    if reuse_ingest:
        ingest = IngestResult()
        print(f"\n[{unit.group_id}] reusing ingested graph (--no-ingest)", flush=True)
    else:
        print(f"\n[{unit.group_id}] ingesting {len(unit.documents)} docs …", flush=True)
        ingest = await ingestor.ingest(session_id, unit.documents, namespace=namespace)
        print(
            f"[{unit.group_id}] ingest done: chunks={ingest.chunks} "
            f"entities={ingest.entities} flush_nodes={ingest.flush_nodes}",
            flush=True,
        )

    q_sem = asyncio.Semaphore(max(1, config.question_concurrency))

    async def _answer_one(idx: int, case: EvalCase) -> tuple[int, CaseResult]:
        async with q_sem:
            probe = await retriever.query(
                session_id, case.question, k=config.probe_k, namespace=namespace
            )
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
        done = await counter.bump()
        preview = answer[:50].replace("\n", " ")
        print(
            f"  [{done:4d}/{counter.total}] {case.case_id}  "
            f"judge={score:.2f}  gen={preview!r}",
            flush=True,
        )
        return idx, result

    answered = await asyncio.gather(
        *[_answer_one(i, c) for i, c in enumerate(unit.cases)]
    )
    results = [r for _, r in sorted(answered, key=lambda t: t[0])]

    if config.purge:
        await client.call_tool(
            "memory_purge",
            {"scope": "both", "session_id": session_id, "namespace": namespace},
        )
    return results


async def drop_namespaces(config: BaseEvalConfig, namespaces: list[str]) -> None:
    """Drop each named namespace (both physical databases) for a clean re-ingest."""
    async with GraphKnowsMCPClient(
        config.command, request_timeout_s=config.request_timeout_s, env=_evaluation_env()
    ) as client:
        for ns in namespaces:
            result = await client.call_tool("memory_drop_namespace", {"namespace": ns})
            log.info("Dropped namespace %s: %s", ns, result)


async def run_units(
    adapter: BenchmarkAdapter,
    config: BaseEvalConfig,
    units: list[IngestUnit],
    *,
    run_id: str,
    reuse_ingest: bool = False,
) -> RunReport:
    """Run all units through the pipeline concurrently and return a report.

    Units (conversations) run up to ``unit_concurrency`` at a time — safe because
    each lives in its own database pair. With ``reuse_ingest`` the ingest+flush
    step is skipped and retrieval runs against the namespace's already-ingested
    graph — the "just tweak retrieval" loop, with no reingest cost.
    """
    total = sum(len(u.cases) for u in units)
    counter = _Progress(total)
    results_by_unit: dict[int, list[CaseResult]] = {}
    ckpt_lock = asyncio.Lock()
    unit_sem = asyncio.Semaphore(max(1, config.unit_concurrency))
    stop = {"flag": False}

    def _ordered() -> list[CaseResult]:
        return [c for i in sorted(results_by_unit) for c in results_by_unit[i]]

    async with GraphKnowsMCPClient(
        config.command,
        request_timeout_s=config.request_timeout_s,
        env=_evaluation_env(),
    ) as client:
        ingestor = Ingestor(client, config)
        retriever = Retriever(client, config)

        async def _worker(idx: int, unit: IngestUnit) -> None:
            if stop["flag"]:
                return
            async with unit_sem:
                if stop["flag"]:
                    return
                try:
                    results = await _run_unit(
                        unit,
                        adapter=adapter,
                        config=config,
                        ingestor=ingestor,
                        retriever=retriever,
                        client=client,
                        counter=counter,
                        reuse_ingest=reuse_ingest,
                    )
                except Exception as exc:
                    stop["flag"] = True
                    if _is_credit_exhaustion(exc):
                        # No traceback: out-of-credits is an operational stop.
                        log.error("Stopping — OpenRouter credits exhausted: %s", exc)  # noqa: TRY400
                    else:
                        log.exception("Stopping — unit %s failed", unit.group_id)
                    return
            async with ckpt_lock:
                results_by_unit[idx] = results
                _checkpoint(run_id, adapter, config, _ordered())
                log.info(
                    "Checkpoint: %d units done, %d cases (last %s)",
                    len(results_by_unit), sum(len(r) for r in results_by_unit.values()),
                    unit.group_id,
                )

        await asyncio.gather(*[_worker(i, u) for i, u in enumerate(units)])

    cases = _ordered()
    return RunReport(
        run_id=run_id,
        benchmark=adapter.name,
        config=config.model_dump(mode="json"),
        cases=cases,
        summary=summarize(cases),
    )
