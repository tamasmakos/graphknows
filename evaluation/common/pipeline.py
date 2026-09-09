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
import re
import time
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
from evaluation.common.lexical import (
    evidence_recall,
    gold_context_coverage,
    gold_max_overlap,
    gold_reachable,
)
from evaluation.common.llm import ChatLLM, CreditExhaustedError, resolve_model
from evaluation.common.reporting import summarize, write_report
from evaluation.common.retrieval import Retriever
from graphknows.integrations.client import GraphKnowsMCPClient

log = logging.getLogger(__name__)

# A refusal to answer. Tracked as its OWN rate rather than folded into accuracy:
# "the memories do not mention X" when the evidence WAS retrieved is a
# grounding/prompt defect, and it is invisible if you only look at the score.
_ABSTAIN_RE = re.compile(
    r"do(es)? not (contain|mention|include|specify|provide)|no (specific |exact )?"
    r"(date|information|mention)|not (mentioned|specified|provided|available|enough)|"
    r"cannot (be )?(answer|determin)|^\s*unknown\s*$|insufficient",
    re.I,
)


class BenchmarkAdapter(abc.ABC):
    """Benchmark-specific behavior: prompts, answer post-processing, judging."""

    name: str = "benchmark"

    def __init__(self, config: BaseEvalConfig, answerer: ChatLLM, judge: DSPyJudge) -> None:
        self.config = config
        self.answerer = answerer
        self.judge = judge

    @abc.abstractmethod
    async def answer(
        self,
        case: EvalCase,
        unit: IngestUnit,
        passages: list[RetrievedPassage],
        facts: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate an answer; return ``(answer, usage_metrics)``.

        ``facts`` is the D8 structured fact-sheet for the query entities (empty
        when disabled); adapters that don't consume it may ignore the argument.
        """

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


# Settings the MCP subprocess needs but does not inherit reliably. The capability
# knobs are here so an A/B can vary one of them per run without touching .env.
_FORWARDED_ENV = (
    "GRAPHKNOWS_ONTOLOGY",
    "GRAPHKNOWS_ONTOLOGY_FILE",  # deprecated alias
    "GRAPHKNOWS_TOPICS",
    "GRAPHKNOWS_DSPY_RELATIONS",
    "GRAPHKNOWS_NEIGHBOR_RADIUS",  # long-document seq-neighbor expansion (BEAM)
)


def _evaluation_env() -> dict[str, str]:
    """Environment overrides handed to the MCP subprocess."""
    return {k: os.environ[k] for k in _FORWARDED_ENV if os.environ.get(k)}


def _evidence_metrics(
    case: EvalCase, probe: list[RetrievedPassage], context: list[RetrievedPassage]
) -> dict[str, float]:
    """Evidence-recall at probe depth and in the generator's context.

    Emits NOTHING when the benchmark ships no evidence for the case, so the
    run summary averages only over cases that have it — no sentinel to strip,
    and benchmarks without evidence simply never show the metric.
    """
    evidence = [str(t) for t in (case.extras.get("evidence") or [])]
    if not evidence:
        return {}
    return {
        "evidence_count": float(len(evidence)),
        "evidence_recall_at_probe": round(evidence_recall(evidence, [p.text for p in probe]), 3),
        "evidence_recall_in_context": round(
            evidence_recall(evidence, [p.text for p in context]), 3
        ),
    }


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
            # The retrieval metric to tune on: did we actually retrieve the gold
            # evidence turns? Per-hop and un-inflatable, unlike the lexical
            # answer-token proxy above. Absent when the benchmark has no evidence.
            **_evidence_metrics(case, probe, context_passages),
            # Fraction of gold tokens present in the passages the GENERATOR saw:
            # high + wrong answer = generation failure, low = retrieval miss.
            "gold_context_coverage": round(gold_context_coverage(case.gold, context_passages), 3),
            "avg_hit_score": _avg_score(passages),
            "graph_hit_count": sum(1 for p in passages if p.is_graph_hit),
            "ingested_chunks": ingest.chunks,
            "ingested_entities": ingest.entities,
            "ingest_elapsed_s": ingest.ingest_elapsed_s,
            "flush_nodes": ingest.flush_nodes,
            "flush_edges": ingest.flush_edges,
            "flush_elapsed_s": ingest.flush_elapsed_s,
            # One metric per stage, so the slow one is named in the report
            # rather than hidden inside the total.
            **{f"flush_stage_{k}_s": v for k, v in ingest.flush_stage_s.items()},
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


def feeding_mode(units: list[IngestUnit]) -> str:
    """How the run fed its data, for the panel report (contracts/panel-report.md).

    ``turn_by_turn`` as soon as any unit carries turns: those units buffer TURN
    rows and let the pipeline's own sliding window chunk them, which is the mode
    the panel row must state.
    """
    turn_fed = any(doc.turns for unit in units for doc in unit.documents)
    return "turn_by_turn" if turn_fed else "document"


async def _ingest_unit(
    unit: IngestUnit,
    *,
    config: BaseEvalConfig,
    ingestor: Ingestor,
    reuse_ingest: bool,
) -> IngestResult:
    """Ingest one conversation into its own namespace (the memory-heavy phase).

    Called while holding an ingest-queue slot, so only ``ingest_workers``
    conversations run the embedding model + NER + vector-index build at once.
    """
    if reuse_ingest:
        print(f"\n[{unit.group_id}] reusing ingested graph (--no-ingest)", flush=True)
        return IngestResult()
    session_id = f"{config.session_prefix}-{unit.group_id}"
    namespace = unit_namespace(config, unit.group_id)
    print(f"\n[{unit.group_id}] ingesting {len(unit.documents)} docs …", flush=True)
    ingest = await ingestor.ingest(session_id, unit.documents, namespace=namespace)
    # ingest_error_count was computed and never surfaced: a run in which EVERY
    # document failed still printed "ingest done: chunks=0" and went on to answer
    # 152 questions against a partially-written graph. A failed document is not a
    # question-level flake — it silently removes evidence from every question — so
    # it is loud, and a total failure is fatal rather than merely reported.
    if ingest.ingest_error_count:
        print(
            f"[{unit.group_id}] !! INGEST ERRORS: {ingest.ingest_error_count}/"
            f"{len(unit.documents)} documents failed — the graph is INCOMPLETE",
            flush=True,
        )
    if ingest.ingest_error_count >= len(unit.documents):
        raise RuntimeError(
            f"[{unit.group_id}] every document failed to ingest "
            f"({ingest.ingest_error_count}/{len(unit.documents)}); "
            f"scoring this run would measure a graph that was never built"
        )
    # A turn-fed unit buffers TURN rows and mints nothing until flush, so
    # reporting only `chunks` would print 0 on a perfectly healthy run and read
    # as a total extraction failure.
    written = (
        f"turns={ingest.turns} flush_chunks={ingest.flush_chunks}"
        if ingest.turns
        else f"chunks={ingest.chunks}"
    )
    # The three slowest flush stages, inline. `flush_elapsed_s` alone cannot say
    # WHICH stage took the time, and waiting for the results file to find out
    # means a 20-minute round trip on every performance question.
    slowest = sorted(ingest.flush_stage_s.items(), key=lambda kv: -kv[1])[:3]
    timing = (
        "  slowest: " + ", ".join(f"{name}={secs:.0f}s" for name, secs in slowest if secs >= 1)
        if slowest
        else ""
    )
    print(
        f"[{unit.group_id}] ingest done: {written} "
        f"entities={ingest.entities} flush_nodes={ingest.flush_nodes} "
        f"errors={ingest.ingest_error_count} flush={ingest.flush_elapsed_s:.0f}s{timing}",
        flush=True,
    )
    return ingest


async def _answer_unit(
    unit: IngestUnit,
    ingest: IngestResult,
    *,
    adapter: BenchmarkAdapter,
    config: BaseEvalConfig,
    retriever: Retriever,
    client: GraphKnowsMCPClient,
    counter: _Progress,
    answer_sem: asyncio.Semaphore,
) -> list[CaseResult]:
    """Answer one conversation's questions (the cheap, parallel phase).

    Runs after the ingest slot is released. Each question is retrieve→generate→
    judge — read-only against the DB with no local models — gated only by the
    *global* ``answer_sem`` shared across all conversations, so total in-flight
    LLM calls stay bounded no matter how many conversations answer at once.
    """
    session_id = f"{config.session_prefix}-{unit.group_id}"
    namespace = unit_namespace(config, unit.group_id)

    async def _answer_one(idx: int, case: EvalCase) -> tuple[int, CaseResult]:
        async with answer_sem:
            try:
                # ONLY retrieval is timed. It is the sole stage this system owns:
                # local graph+vector work with no LLM in it. Generation and
                # judging are whatever model happens to be configured, so timing
                # them would report the provider's speed under our name — a number
                # that changes when you swap models and says nothing about the
                # memory layer. Their COST is still recorded (see _operational),
                # because that is what a user actually spends.
                t0 = time.perf_counter()
                probe, facts = await retriever.query(
                    session_id, case.question, k=config.probe_k, namespace=namespace
                )
                retrieve_ms = (time.perf_counter() - t0) * 1000.0

                passages = probe[: config.top_k]
                context_passages = probe[: config.gen_context_k]
                answer, gen_usage = await adapter.answer(case, unit, context_passages, facts)
                score, judge_metrics = await adapter.judge_case(case, answer)

                lexical = adapter.lexical_metrics(case, answer)
                lexical = {
                    **lexical,
                    "retrieve_ms": round(retrieve_ms, 1),
                    "fact_count": len(facts),
                    "context_passage_count": len(context_passages),
                    # Refusing to answer is a distinct failure from answering
                    # wrongly — it needs a prompt/grounding fix, not a retrieval
                    # one — so it is counted rather than hidden inside accuracy.
                    "abstained": bool(_ABSTAIN_RE.search(answer or "")),
                }
            except Exception as exc:
                # Credit exhaustion is the ONLY hard stop — re-raise it. Any other
                # per-question failure (a transient empty completion, a provider
                # 4xx, a flaky judge) is scored 0 and the run continues; a single
                # bad question must never halt a 1540-question run.
                if _is_credit_exhaustion(exc):
                    raise
                log.warning("Question %s failed (scored 0.0): %s", case.case_id, exc)
                done = await counter.bump()
                print(
                    f"  [{done:4d}/{counter.total}] {case.case_id}  judge=0.00  ERROR", flush=True
                )
                return idx, build_case_result(
                    case,
                    config=config,
                    session_id=session_id,
                    passages=[],
                    probe=[],
                    context_passages=[],
                    generated_answer="",
                    score=0.0,
                    judge_metrics={"judge_label": "ERROR", "judge_reasoning": str(exc)[:200]},
                    lexical={},
                    gen_usage={},
                    ingest=ingest,
                )
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
            f"  [{done:4d}/{counter.total}] {case.case_id}  judge={score:.2f}  gen={preview!r}",
            flush=True,
        )
        return idx, result

    answered = await asyncio.gather(*[_answer_one(i, c) for i, c in enumerate(unit.cases)])
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
    """Run all units through the pipeline and return a report.

    Two decoupled stages per conversation, so each is bounded by the resource it
    actually stresses:

    * **Ingest** — a bounded queue: at most ``ingest_workers`` conversations hold
      an ``ingest_sem`` slot and run the memory-heavy embedding/NER/vector build
      at once. The rest wait their turn, keeping the MCP-server process off the
      OOM line.
    * **Answer** — parallel: as soon as a conversation finishes ingest it releases
      its slot and its questions answer concurrently, gated only by the *global*
      ``question_concurrency`` cap on total in-flight LLM calls. Conversations
      that finished ingest answer while later ones are still queued to ingest.

    With ``reuse_ingest`` the ingest stage is a no-op and everything is answering.
    """
    total = sum(len(u.cases) for u in units)
    counter = _Progress(total)
    results_by_unit: dict[int, list[CaseResult]] = {}
    ckpt_lock = asyncio.Lock()
    ingest_sem = asyncio.Semaphore(max(1, config.ingest_workers))
    answer_sem = asyncio.Semaphore(max(1, config.question_concurrency))
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
            try:
                # Stage 1 — bounded ingest queue (memory-heavy, capped).
                async with ingest_sem:
                    if stop["flag"]:
                        return
                    ingest = await _ingest_unit(
                        unit, config=config, ingestor=ingestor, reuse_ingest=reuse_ingest
                    )
                # Stage 2 — parallel answering (cheap, global LLM cap). The ingest
                # slot is already released, so a queued conversation can ingest now.
                results = await _answer_unit(
                    unit,
                    ingest,
                    adapter=adapter,
                    config=config,
                    retriever=retriever,
                    client=client,
                    counter=counter,
                    answer_sem=answer_sem,
                )
            except Exception as exc:
                if _is_credit_exhaustion(exc):
                    # Credits out is the only hard stop — no point continuing.
                    stop["flag"] = True
                    log.error("Stopping — OpenRouter credits exhausted: %s", exc)  # noqa: TRY400
                else:
                    # One conversation failing (e.g. its ingest errored) skips that
                    # conversation; the rest of the run continues.
                    log.exception("Skipping unit %s — failed", unit.group_id)
                return
            async with ckpt_lock:
                results_by_unit[idx] = results
                _checkpoint(run_id, adapter, config, _ordered())
                log.info(
                    "Checkpoint: %d units done, %d cases (last %s)",
                    len(results_by_unit),
                    sum(len(r) for r in results_by_unit.values()),
                    unit.group_id,
                )

        await asyncio.gather(*[_worker(i, u) for i, u in enumerate(units)])

    cases = _ordered()
    return RunReport(
        run_id=run_id,
        benchmark=adapter.name,
        config=config.model_dump(mode="json"),
        cases=cases,
        summary=summarize(cases) | {"feeding_mode": feeding_mode(units)},
    )
