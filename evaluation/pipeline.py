"""The LoCoMo evaluation pipeline.

One workflow, run per dialogue: ingest the dialogue's sessions once, then for
each question retrieve → generate → score (symmetric token F1 vs gold). See
:mod:`evaluation.config` for the single hardcoded configuration.
"""

from __future__ import annotations

import logging
import os

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
from evaluation.reporting import summarize
from evaluation.retrieval import Retriever
from evaluation.scoring import gold_max_overlap, gold_reachable, token_f1

log = logging.getLogger(__name__)


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
    ingest: IngestResult,
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
    client: GraphknowsMCPClient,
    progress: tuple[int, int],
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
            passages = await retriever.query(session_id, case.question, k=config.top_k)
            probe = await retriever.query(session_id, case.question, k=config.probe_k)
            answer = await generator.generate(case.question, probe[: config.gen_context_k])

            result = _build_case_result(
                case,
                config=config,
                session_id=session_id,
                passages=passages,
                probe=probe,
                generated_answer=answer,
                ingest=ingest,
            )
            results.append(result)

            case_idx += 1
            preview = answer[:50].replace("\n", " ")
            print(
                f"  [{case_idx:3d}/{total}] {case.case_id}  F1={result.score:.3f}  gen={preview!r}",
                flush=True,
            )
    finally:
        if config.purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    return results


async def run_evaluation(config: EvalConfig, *, run_id: str, limit: int, offset: int) -> RunReport:
    """Run the LoCoMo evaluation and return a populated :class:`RunReport`."""
    generator = AnswerGenerator(model=config.generation_model or None)
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

        for dialogue in dialogues:
            results = await _run_dialogue(
                dialogue,
                config=config,
                run_id=run_id,
                ingestor=ingestor,
                retriever=retriever,
                generator=generator,
                client=client,
                progress=(len(cases), total),
            )
            cases.extend(results)

    return RunReport(run_id=run_id, config=config, cases=cases, summary=summarize(cases))
