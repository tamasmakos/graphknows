"""Ingestion stage: ingest an ingest unit's documents, optionally flush to LTM.

Wraps the ``memory_ingest`` / ``memory_flush`` / ``ltm_entities`` MCP tools and
returns a typed :class:`IngestResult` aggregating chunk/entity/graph counts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import EvalDocument, IngestResult
from graphknows.integrations.client import GraphKnowsMCPClient

log = logging.getLogger(__name__)


class Ingestor:
    """Ingests one unit's documents into one memory session."""

    def __init__(self, client: GraphKnowsMCPClient, config: BaseEvalConfig) -> None:
        self._client = client
        self._config = config

    async def ingest(
        self, session_id: str, documents: list[EvalDocument], *, namespace: str = ""
    ) -> IngestResult:
        """Ingest all documents (concurrency opt-in), then optionally flush to LTM.

        Documents of ONE unit share that unit's namespace graph, so parallel MERGE
        of a shared entity name still races the ENTITY(name) UNIQUE index within
        the unit — keep EVAL_INGEST_CONCURRENCY at 1. Cross-unit parallelism is
        safe because each unit is a physically separate database pair, and is
        driven by the pipeline (EVAL_UNIT_CONCURRENCY), not here.
        """
        cap = int(os.environ.get("EVAL_INGEST_CONCURRENCY", "1"))
        sem = asyncio.Semaphore(cap)

        # EVAL_INFER=0 ingests chunks + embeddings only, skipping GLiNER
        # entity/relation extraction (and its per-relation graph writes) — the
        # ingest cost that dominates but never touches the vector index.
        infer = os.environ.get("EVAL_INFER", "1") != "0"

        async def _ingest_one(index: int, doc: EvalDocument) -> dict[str, Any]:
            title = doc.title or f"{session_id}-doc-{index}"
            # Turn-fed units send `messages`, which buffers cheap TURN rows that
            # the flush below drains through the real pipeline — the same
            # lifecycle a live agent has, and the one whose chunking is the
            # sliding window. A document sends `text` and is extracted eagerly.
            payload: dict[str, Any] = {
                "session_id": session_id,
                "title": title,
                "infer": infer,
                "metadata": {"anchor_date": doc.anchor_date},
                "namespace": namespace,
            }
            payload["messages" if doc.turns else "text"] = doc.turns or doc.text
            async with sem:
                try:
                    result = await self._client.call_tool("memory_ingest", payload)
                except Exception as exc:
                    log.exception("INGEST FAILED doc=%s (%s)", title, type(exc).__name__)
                    print(
                        f"  !! ingest failed: doc={title} {type(exc).__name__}: {str(exc)[:400]}",
                        flush=True,
                    )
                    return {"error": str(exc)[:400]}
            out = result if isinstance(result, dict) else {"result": result}
            # A failed tool call comes back as {"text": "<error>"} (see
            # normalize_tool_result) — the reason was RIGHT THERE and discarded,
            # so a document that dropped its whole chunk set looked identical to
            # one that succeeded. Extraction failures silently delete evidence
            # from every question; never swallow them.
            if "chunks" not in out:
                detail = out.get("text") or out.get("error") or str(out)[:400]
                log.error("INGEST FAILED doc=%s: %s", title, detail)
                print(f"  !! ingest failed: doc={title} :: {str(detail)[:400]}", flush=True)
            return out

        outputs: list[dict[str, Any]] = await asyncio.gather(
            *[_ingest_one(i, d) for i, d in enumerate(documents)]
        )

        ingest = IngestResult(
            documents=len(outputs),
            chunks=sum(int(o.get("chunks", 0)) for o in outputs),
            entities=sum(int(o.get("entities", 0)) for o in outputs),
            turns=sum(int(o.get("turns", 0)) for o in outputs),
            ingest_elapsed_s=round(sum(float(o.get("elapsed_s", 0)) for o in outputs), 3),
            ingest_error_count=sum(1 for o in outputs if "chunks" not in o),
        )

        if self._config.promote_to_ltm:
            await self._flush(session_id, ingest, namespace=namespace)
        return ingest

    async def _flush(self, session_id: str, ingest: IngestResult, *, namespace: str = "") -> None:
        """Flush the session to LTM and capture graph + entity counts in-place."""
        try:
            result = await self._client.call_tool(
                "memory_flush", {"session_id": session_id, "namespace": namespace}
            )
            flush = result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            # `%s` of a TimeoutError is the EMPTY STRING, so this line used to
            # read "flush failed for <sid>: " and looked like a truncated log
            # rather than the timeout it was. Name the type — the run is scored
            # either way, and a scored half-flushed graph is the worst outcome
            # here (it looks like a real, bad number).
            log.warning(
                "flush failed for %s: %s: %s (the conversation will be scored on an "
                "INCOMPLETE graph — treat its metrics as invalid)",
                session_id,
                type(exc).__name__,
                exc or "<no message>",
            )
            ingest.flush_error_count = 1
            return

        ingest.flush_nodes = int(flush.get("nodes", 0))
        ingest.flush_edges = int(flush.get("edges", 0))
        ingest.flush_chunks = int(flush.get("chunks", 0))
        ingest.flush_elapsed_s = round(float(flush.get("elapsed_s", 0)), 3)
        ingest.flush_error_count = len(flush.get("errors", []))
        ingest.flush_stage_s = {str(k): float(v) for k, v in (flush.get("stage_s") or {}).items()}
