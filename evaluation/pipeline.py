"""The LoCoMo evaluation pipeline.

One workflow, run per dialogue: ingest the dialogue's sessions once, then for
each question retrieve → generate → score (symmetric token F1 vs gold). See
:mod:`evaluation.config` for the single hardcoded configuration.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

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

# ── Ingest-reuse cache (opt-in via EVAL_REUSE_INGEST=1) ──────────────────────


def _ingest_config_hash(config: EvalConfig) -> str:
    """Stable SHA-256 fingerprint of the ingest-affecting config fields."""
    payload = json.dumps(
        {
            # Real mode comes from the MODE env (set by --mode); config.relation_
            # extraction_mode is the hardcoded eval default and does NOT vary by --mode.
            "mode": os.environ.get("MODE") or config.relation_extraction_mode,
            "max_chars_per_document": config.max_chars_per_document,
            "max_documents": config.max_documents,
            "promote_to_ltm": config.promote_to_ltm,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_ingest_marker(marker_path: Path) -> dict[str, str]:
    """Return the persisted cache marker, or an empty dict if missing/corrupt."""
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_ingest_marker(marker_path: Path, marker: dict[str, str]) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")


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
    env = {"RELATION_EXTRACTION_MODE": config.relation_extraction_mode}
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
            "relation_extraction_mode": config.relation_extraction_mode,
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
    # Ingest-reuse cache params (set by run_evaluation when EVAL_REUSE_INGEST=1).
    # When None/False the function behaves exactly as before.
    cache_session_id: str | None = None,
    cache_hit: bool = False,
    should_purge: bool | None = None,
) -> list[CaseResult]:
    """Ingest one dialogue, then retrieve+generate+score each of its questions."""
    case_idx, total = progress

    # Determine the effective session_id and purge behaviour.
    # When REUSE is off (cache_session_id is None), these are the pre-existing values.
    if cache_session_id is not None:
        session_id = cache_session_id
        purge_after = False  # keep cached data in the database
    else:
        session_id = f"{config.session_prefix}-{run_id}-{dialogue.dialogue_id}"
        purge_after = config.purge if should_purge is None else should_purge

    if cache_hit:
        log.info("ingest cache HIT: %s", session_id)
        ingest = IngestResult()  # zero-cost placeholder; data already in DB
    else:
        print(f"\n[{dialogue.dialogue_id}] ingesting {len(dialogue.documents)} docs …", flush=True)

    try:
        if not cache_hit:
            ingest = await ingestor.ingest(session_id, dialogue.documents)
            print(
                f"[{dialogue.dialogue_id}] ingest done: chunks={ingest.chunks} "
                f"entities={ingest.entities} flush_nodes={ingest.flush_nodes}",
                flush=True,
            )

        # Default 1 (sequential). The generator/judge now use DSPy native async
        # (acall), so raising EVAL_CONCURRENCY is correct-by-design — but validate a
        # sample (mean-F1 stable vs sequential) before trusting an unattended run.
        concurrency = int(os.environ.get("EVAL_CONCURRENCY", "1"))
        sem = asyncio.Semaphore(concurrency)
        results: list[CaseResult | None] = [None] * len(dialogue.cases)
        completed = 0

        async def _process_case(idx: int, case: LocomoCase) -> None:
            nonlocal completed
            async with sem:
                # One retrieval at probe depth; the scored top_k passages are its head.
                # (A second k=top_k query would re-run the whole retriever — expensive
                # for the agentic ReAct path — for an identically-ranked prefix.)
                probe = await retriever.query(session_id, case.question, k=config.probe_k)
                passages = probe[: config.top_k]
                context_passages = probe[: config.gen_context_k]
                answer, gen_usage = await generator.generate(case.question, context_passages)
                judge_correct = (
                    await judge.judge(case.question, case.gold, answer)
                    if judge is not None
                    else None
                )

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
                results[idx] = result

                completed += 1
                counter = case_idx + completed
                preview = answer[:50].replace("\n", " ")
                print(
                    f"  [{counter:3d}/{total}] {case.case_id}  F1={result.score:.3f}  "
                    f"judge={'Y' if judge_correct else 'N'}  gen={preview!r}",
                    flush=True,
                )
                # Intra-dialogue checkpoint: large dialogues (conv-26 has 152 cases)
                # would otherwise only persist at the end. Save every 30 cases so an
                # unexpected crash never loses a long dialogue's progress.
                if on_progress is not None and completed % 30 == 0:
                    on_progress([r for r in results if r is not None])

        await asyncio.gather(*[_process_case(i, c) for i, c in enumerate(dialogue.cases)])

    finally:
        if purge_after:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return results  # type: ignore[return-value]  # all slots filled by gather


async def run_evaluation(config: EvalConfig, *, run_id: str, limit: int, offset: int) -> RunReport:
    """Run the LoCoMo evaluation and return a populated :class:`RunReport`."""
    generator = AnswerGenerator(model=config.generation_model or None)
    # LLM-as-judge is a SECONDARY semantic metric; token-F1 is the primary score.
    # It costs ~3 LLM calls per question (the bulk of eval request volume), so it
    # is OFF by default to conserve credits — enable with EVAL_JUDGE=1.
    judge = (
        LLMJudge(model=config.generation_model or None)
        if os.environ.get("EVAL_JUDGE") == "1"
        else None
    )
    # Optional inline DSPy semantic scoring (SemanticF1 + CompleteAndGrounded).
    # Off by default; enable with EVAL_DSPY_METRICS=1 to score each case inline.
    dspy_scorer = None
    if os.environ.get("EVAL_DSPY_METRICS"):
        from evaluation.scoring.dspy_metrics import DspyMetricScorer

        dspy_scorer = DspyMetricScorer(model=config.generation_model or None)
        log.info("Inline DSPy metrics enabled (SemanticF1 + CompleteAndGrounded)")
    dialogues = load_dialogues(config, limit=limit, offset=offset)
    total = sum(len(d.cases) for d in dialogues)

    # Ingest-reuse cache: opt-in via EVAL_REUSE_INGEST=1.
    # When off, every variable below is unused and behaviour is byte-identical.
    reuse = os.environ.get("EVAL_REUSE_INGEST") == "1"
    if reuse:
        config_hash = _ingest_config_hash(config)
        marker_path = Path(config.output_dir) / ".ingest_cache.json"
        marker = _load_ingest_marker(marker_path)
        log.info("Ingest-reuse cache enabled (config_hash=%s, marker=%s)", config_hash, marker_path)

    cases: list[CaseResult] = []
    async with GraphknowsMCPClient(
        config.command,
        request_timeout_s=config.request_timeout_s,
        env=_evaluation_env(config),
    ) as client:
        if config.pre_run_purge_ltm and not reuse:
            # Skip the LTM wipe when reuse is on — it would delete the cache.
            log.info("Purging all LTM before the run (pre_run_purge_ltm=True)")
            await client.call_tool("memory_purge", {"scope": "ltm", "session_id": ""})

        ingestor = Ingestor(client, config)
        retriever = Retriever(client, config)

        for i, dialogue in enumerate(dialogues, 1):
            # Checkpoint prior dialogues + this dialogue's progress-so-far.
            def _progress_ckpt(partial: list[CaseResult], _done: list[CaseResult] = cases) -> None:
                _checkpoint(run_id, config, _done + partial)

            # Compute cache decision for this dialogue (no-op when reuse is off).
            extra_kwargs: dict = {}
            if reuse:
                mode = os.environ.get("MODE") or config.relation_extraction_mode
                csid = f"eval-cache-{mode}-{dialogue.dialogue_id}"
                hit = marker.get(csid) == config_hash
                extra_kwargs = {"cache_session_id": csid, "cache_hit": hit}
                if not hit:
                    log.info("ingest cache MISS, will populate: %s", csid)

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
                    **extra_kwargs,
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

            # After a successful MISS, record the cache entry.
            if reuse and not extra_kwargs.get("cache_hit"):
                csid = extra_kwargs["cache_session_id"]
                marker[csid] = config_hash
                _save_ingest_marker(marker_path, marker)
                log.info("ingest cache MISS, populated: %s", csid)

            cases.extend(results)
            _checkpoint(run_id, config, cases)
            log.info("Checkpoint: %d cases after dialogue %s (%d/%d)",
                     len(cases), dialogue.dialogue_id, i, len(dialogues))

    return RunReport(run_id=run_id, config=config, cases=cases, summary=summarize(cases))
