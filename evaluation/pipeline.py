"""The LoCoMo evaluation pipeline.

One workflow, run per dialogue: ingest the dialogue's sessions once, then for
each question retrieve → generate → score (symmetric token F1 vs gold). See
:mod:`evaluation.config` for the single hardcoded configuration.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from evaluation.client import GraphknowsMCPClient
from evaluation.config import EvalConfig
from evaluation.datamodels import (
    CaseResult,
    Dialogue,
    IngestResult,
    LocomoCase,
    RunReport,
)
from evaluation.dataset import load_dialogues
from evaluation.generation import AnswerGenerator
from evaluation.ingestion import Ingestor
from evaluation.reporting import summarize, write_report
from evaluation.retrieval import Retriever
from evaluation.scoring import LLMJudge, gold_max_overlap, gold_reachable, token_f1

log = logging.getLogger(__name__)


def _is_credit_exhaustion(exc: Exception) -> bool:
    """True when *exc* is an OpenRouter out-of-credits error (vs a real bug)."""
    msg = str(exc).lower()
    return "credit" in msg or "402" in msg or "payment" in msg


def _checkpoint(run_id: str, config: EvalConfig, cases: list[CaseResult]) -> None:
    """Persist results so far; a later crash never loses completed dialogues."""
    try:
        write_report(RunReport(run_id=run_id, config=config, cases=list(cases), summary=summarize(cases)))
    except Exception as exc:  # noqa: BLE001 — checkpointing must never break the run
        log.warning("Checkpoint write failed (non-fatal): %s", exc)


def _evaluation_env(config: EvalConfig) -> dict[str, str]:
    """Environment overrides handed to the MCP subprocess."""
    env: dict[str, str] = {}
    # Propagate the ontology file only when already set; otherwise the extractor
    # uses its open-domain relation-type defaults.
    if os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE"):
        env["GRAPHKNOWS_ONTOLOGY_FILE"] = os.environ["GRAPHKNOWS_ONTOLOGY_FILE"]
    return env


def _avg_score(passages: list) -> float:
    scores = [p.score for p in passages if isinstance(p.score, (int, float))]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _build_case_result(
    case: LocomoCase,
    *,
    config: EvalConfig,
    session_id: str,
    passages: list,
    probe: list,
    generated_answer: str,
    gen_usage: dict,
    judge_correct: bool,
    ingest: IngestResult,
    dspy_metrics: dict | None = None,
) -> CaseResult:
    score = token_f1(case.gold, generated_answer)
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        gold=case.gold,
        category=case.category,
        dialogue_id=case.dialogue_id,
        passages=passages,
        generated_answer=generated_answer,
        score=score,
        metrics={
            "score_type": "token_f1",
            "judge_correct": judge_correct,
            "gold_max_token_overlap": round(gold_max_overlap(case.gold, probe), 3),
            "gold_reachable_at_20": gold_reachable(case.gold, probe),
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
            **(dspy_metrics or {}),
        },
        metadata={
            "session_id": session_id,
            "recall_scope": config.recall_scope,
            "promote_to_ltm": config.promote_to_ltm,
            "top_k": config.top_k,
        },
    )


async def _run_dialogue(
    dialogue: Dialogue,
    *,
    config: EvalConfig,
    run_id: str,
    ingestor: Ingestor,
    retriever: Retriever,
    generator: AnswerGenerator,
    judge: LLMJudge,
    client: GraphknowsMCPClient,
    progress: tuple[int, int],
    dspy_scorer: object | None = None,
    on_progress: "Callable[[list[CaseResult]], None] | None" = None,
) -> list[CaseResult]:
    """Ingest one dialogue, then retrieve+generate+score each of its questions."""
    case_idx, total = progress
    session_id = f"{config.session_prefix}-{run_id}-{dialogue.dialogue_id}"
    results: list[CaseResult] = []

    print(f"\n[{dialogue.dialogue_id}] ingesting {len(dialogue.documents)} docs …", flush=True)
    try:
        ingest = await ingestor.ingest(session_id, dialogue.documents)
        print(
            f"[{dialogue.dialogue_id}] ingest done: chunks={ingest.chunks} "
            f"entities={ingest.entities} flush_nodes={ingest.flush_nodes}",
            flush=True,
        )

        for case in dialogue.cases:
            # One retrieval at probe depth; the scored top_k passages are its head.
            # (A second k=top_k query would re-run the whole retriever — expensive
            # for the agentic ReAct path — for an identically-ranked prefix.)
            probe = await retriever.query(session_id, case.question, k=config.probe_k)
            passages = probe[: config.top_k]
            context_passages = probe[: config.gen_context_k]
            answer, gen_usage = await generator.generate(case.question, context_passages)
            judge_correct = await judge.judge(case.question, case.gold, answer)

            dspy_metrics: dict | None = None
            if dspy_scorer is not None:
                context = "\n\n".join(p.text for p in context_passages)
                scores = await dspy_scorer.score(case.question, case.gold, answer, context)
                dspy_metrics = scores.as_metrics()

            result = _build_case_result(
                case,
                config=config,
                session_id=session_id,
                passages=passages,
                probe=probe,
                generated_answer=answer,
                gen_usage=gen_usage,
                judge_correct=judge_correct,
                ingest=ingest,
                dspy_metrics=dspy_metrics,
            )
            results.append(result)

            case_idx += 1
            preview = answer[:50].replace("\n", " ")
            print(
                f"  [{case_idx:3d}/{total}] {case.case_id}  F1={result.score:.3f}  "
                f"judge={'Y' if judge_correct else 'N'}  gen={preview!r}",
                flush=True,
            )
            # Intra-dialogue checkpoint: large dialogues (conv-26 has 152 cases)
            # would otherwise only persist at the end. Save every 30 cases so an
            # unexpected crash never loses a long dialogue's progress.
            if on_progress is not None and len(results) % 30 == 0:
                on_progress(results)
    finally:
        if config.purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return results


async def run_evaluation(config: EvalConfig, *, run_id: str, limit: int, offset: int) -> RunReport:
    """Run the LoCoMo evaluation and return a populated :class:`RunReport`."""
    generator = AnswerGenerator(model=config.generation_model or None)
    judge = LLMJudge(model=config.generation_model or None)
    # Optional inline DSPy semantic scoring (SemanticF1 + CompleteAndGrounded).
    # Off by default; enable with EVAL_DSPY_METRICS=1 to score each case inline.
    dspy_scorer = None
    if os.environ.get("EVAL_DSPY_METRICS"):
        from evaluation.scoring.dspy_metrics import DspyMetricScorer

        dspy_scorer = DspyMetricScorer(model=config.generation_model or None)
        log.info("Inline DSPy metrics enabled (SemanticF1 + CompleteAndGrounded)")
    dialogues = load_dialogues(config, limit=limit, offset=offset)
    total = sum(len(d.cases) for d in dialogues)

    cases: list[CaseResult] = []
    async with GraphknowsMCPClient(
        config.command,
        request_timeout_s=config.request_timeout_s,
        env=_evaluation_env(config),
    ) as client:
        if config.pre_run_purge_ltm:
            log.info("Purging all LTM before the run (pre_run_purge_ltm=True)")
            await client.call_tool("memory_purge", {"scope": "ltm", "session_id": ""})

        ingestor = Ingestor(client, config)
        retriever = Retriever(client, config)

        for i, dialogue in enumerate(dialogues, 1):
            # Checkpoint prior dialogues + this dialogue's progress-so-far.
            def _progress_ckpt(partial: list[CaseResult], _done: list[CaseResult] = cases) -> None:
                _checkpoint(run_id, config, _done + partial)

            try:
                results = await _run_dialogue(
                    dialogue,
                    config=config,
                    run_id=run_id,
                    ingestor=ingestor,
                    retriever=retriever,
                    generator=generator,
                    judge=judge,
                    client=client,
                    progress=(len(cases), total),
                    dspy_scorer=dspy_scorer,
                    on_progress=_progress_ckpt,
                )
            except Exception as exc:  # noqa: BLE001 — unattended runs must end with a valid report
                # Any dialogue-level failure (out of credits, MCP timeout, store
                # error) stops the run cleanly with whatever is already checkpointed,
                # rather than crashing a multi-hour run with an unhandled exception.
                if _is_credit_exhaustion(exc):
                    log.error("Stopping after %d/%d dialogues — OpenRouter credits exhausted: %s",
                              i - 1, len(dialogues), exc)
                else:
                    log.exception("Stopping after %d/%d dialogues — dialogue %s failed: %s",
                                  i - 1, len(dialogues), dialogue.dialogue_id, exc)
                break
            cases.extend(results)
            _checkpoint(run_id, config, cases)
            log.info("Checkpoint: %d cases after dialogue %s (%d/%d)",
                     len(cases), dialogue.dialogue_id, i, len(dialogues))

    return RunReport(run_id=run_id, config=config, cases=cases, summary=summarize(cases))
