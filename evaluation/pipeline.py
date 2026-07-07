"""The LoCoMo evaluation pipeline.

One workflow, run per dialogue: ingest the dialogue's sessions once, then for
each question retrieve → generate → score with DSPy SemanticF1 (headline) and
token-F1 (diagnostic). See :mod:`evaluation.config` for configuration.
"""

from __future__ import annotations

import logging
import os
import re
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
from evaluation.ingestion import Ingestor
from evaluation.reporting import summarize, write_report
from evaluation.retrieval import Retriever
from evaluation.scoring import gold_max_overlap, gold_reachable, locomo_f1, token_f1

# Deferred imports for dspy-dependent modules so that importing evaluation.pipeline
# for its non-dspy helpers (_build_case_result etc.) works even when dspy is absent.
# The actual import happens inside run_evaluation() / _run_dialogue() at call time.
TYPE_CHECKING = False
if TYPE_CHECKING:  # pragma: no cover
    from evaluation.generation import AnswerGenerator
    from evaluation.scoring.dspy_metrics import DspyMetricScorer

log = logging.getLogger(__name__)


def select_gen_context_k(config: EvalConfig, mode: str | None) -> int:
    """Return the number of passages to feed the generator based on run mode."""
    return config.gen_context_k_free if mode == "llm_free" else config.gen_context_k


def _is_credit_exhaustion(exc: Exception) -> bool:
    """True when *exc* is an OpenRouter out-of-credits error (vs a real bug)."""
    msg = str(exc).lower()
    return "credit" in msg or "402" in msg or "payment" in msg


def _checkpoint(run_id: str, config: EvalConfig, cases: list[CaseResult]) -> None:
    """Persist results so far; a later crash never loses completed dialogues."""
    try:
        write_report(
            RunReport(run_id=run_id, config=config, cases=list(cases), summary=summarize(cases))
        )
    except Exception as exc:
        log.warning("Checkpoint write failed (non-fatal): %s", exc)


def _evaluation_env(config: EvalConfig) -> dict[str, str]:
    """Environment overrides handed to the MCP subprocess."""
    env: dict[str, str] = {}
    if os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE"):
        env["GRAPHKNOWS_ONTOLOGY_FILE"] = os.environ["GRAPHKNOWS_ONTOLOGY_FILE"]
    return env


def _avg_score(passages: list) -> float:
    scores = [p.score for p in passages if isinstance(p.score, (int, float))]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


_WORD_RE = re.compile(r"[a-z0-9']+")


def _gold_context_coverage(gold: str, passages: list) -> float:
    """Fraction of the gold's content tokens present in the generator's passages.

    High coverage + wrong answer = generation failure; low coverage = retrieval
    miss. Splitting the two per-case makes failure attribution readable straight
    from the results JSON.
    """
    gold_toks = {w for w in _WORD_RE.findall((gold or "").lower()) if len(w) > 2}
    if not gold_toks:
        return 0.0
    ctx = " ".join(p.text for p in passages).lower()
    ctx_toks = set(_WORD_RE.findall(ctx))
    return len(gold_toks & ctx_toks) / len(gold_toks)


def _build_case_result(
    case: LocomoCase,
    *,
    config: EvalConfig,
    session_id: str,
    passages: list,
    probe: list,
    generated_answer: str,
    gen_usage: dict,
    ingest: IngestResult,
    semantic_f1: float = 0.0,
) -> CaseResult:
    tok_f1 = token_f1(case.gold, generated_answer)
    loco_f1 = locomo_f1(case.gold, generated_answer, case.category)
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        gold=case.gold,
        category=case.category,
        dialogue_id=case.dialogue_id,
        passages=passages,
        generated_answer=generated_answer,
        score=semantic_f1,  # headline: SemanticF1
        metrics={
            "score_type": "semantic_f1",
            "semantic_f1": semantic_f1,
            # Official LoCoMo token-F1 (category-aware, stemmed) — paper-comparable.
            "locomo_f1": loco_f1,
            "token_f1": tok_f1,  # plain (non-category) stemmed token-F1
            "gold_max_token_overlap": round(gold_max_overlap(case.gold, probe), 3),
            "gold_reachable_at_20": gold_reachable(case.gold, probe),
            # Fraction of gold tokens present in the passages the GENERATOR saw:
            # high + wrong answer = generation failure, low = retrieval miss.
            "gold_context_coverage": round(_gold_context_coverage(case.gold, passages), 3),
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


async def _run_dialogue(
    dialogue: Dialogue,
    *,
    config: EvalConfig,
    run_id: str,
    ingestor: Ingestor,
    retriever: Retriever,
    generator: AnswerGenerator,
    dspy_scorer: DspyMetricScorer,
    client: GraphknowsMCPClient,
    progress: tuple[int, int],
    on_progress: Callable[[list[CaseResult]], None] | None = None,
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
            probe = await retriever.query(session_id, case.question, k=config.probe_k)
            passages = probe[: config.top_k]
            # passage count fed to generator is mode-specific (llm_free=50, llm_assisted=25)
            context_passages = probe[
                : select_gen_context_k(config, os.environ.get("GRAPHKNOWS_MODE"))
            ]
            answer, gen_usage = await generator.generate(case.question, context_passages)

            # SemanticF1 is the headline metric — always computed.
            scores = await dspy_scorer.score(case.question, case.gold, answer)
            semantic_f1 = scores.semantic_f1

            result = _build_case_result(
                case,
                config=config,
                session_id=session_id,
                passages=passages,
                probe=probe,
                generated_answer=answer,
                gen_usage=gen_usage,
                ingest=ingest,
                semantic_f1=semantic_f1,
            )
            results.append(result)

            case_idx += 1
            preview = answer[:50].replace("\n", " ")
            tok_f1_val = result.metrics.get("token_f1", 0.0)
            print(
                f"  [{case_idx:3d}/{total}] {case.case_id}  "
                f"semF1={semantic_f1:.3f}  tokF1={tok_f1_val:.3f}  gen={preview!r}",
                flush=True,
            )
            if on_progress is not None and len(results) % 30 == 0:
                on_progress(results)
    finally:
        if config.purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return results


async def run_evaluation(config: EvalConfig, *, run_id: str, limit: int, offset: int) -> RunReport:
    """Run the LoCoMo evaluation and return a populated :class:`RunReport`."""
    from evaluation.generation import AnswerGenerator
    from evaluation.scoring.dspy_metrics import DspyMetricScorer

    generator = AnswerGenerator(model=config.generation_model or None)
    dspy_scorer = DspyMetricScorer(model=config.generation_model or None)
    log.info("SemanticF1 scorer ready (headline metric)")
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
                    dspy_scorer=dspy_scorer,
                    client=client,
                    progress=(len(cases), total),
                    on_progress=_progress_ckpt,
                )
            except Exception as exc:
                if _is_credit_exhaustion(exc):
                    log.error(
                        "Stopping after %d/%d dialogues — OpenRouter credits exhausted: %s",
                        i - 1,
                        len(dialogues),
                        exc,
                    )
                else:
                    log.exception(
                        "Stopping after %d/%d dialogues — dialogue %s failed: %s",
                        i - 1,
                        len(dialogues),
                        dialogue.dialogue_id,
                        exc,
                    )
                break
            cases.extend(results)
            _checkpoint(run_id, config, cases)
            log.info(
                "Checkpoint: %d cases after dialogue %s (%d/%d)",
                len(cases),
                dialogue.dialogue_id,
                i,
                len(dialogues),
            )

    return RunReport(run_id=run_id, config=config, cases=cases, summary=summarize(cases))
