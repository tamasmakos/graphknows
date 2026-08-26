"""Langfuse tracing for the eval harnesses — off unless LANGFUSE_* is set.

Deliberately a copy of the ten lines in ``examples/langgraph_agent.py`` rather
than a shared import: that file is meant to be read and pasted whole by someone
integrating graphknows, and an import into ``evaluation`` would break it as an
example. This one exists for the harnesses, which need the extra piece an
example does not — :func:`retrieval_span`, for looking at what recall returned
when the answer was wrong.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


def tracer(session: str, run: str, tags: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    """``(langgraph run config, langfuse client)``, or ``({}, None)`` when unset."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return {}, None
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    config = {
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_session_id": session,
            "langfuse_tags": ["locomo", *(tags or [])],
            "run": run,
        },
    }
    return config, get_client()


@contextmanager
def retrieval_span(langfuse: Any, question: dict[str, Any], session: str) -> Any:
    """Wrap one question's recall so the retrieved set is inspectable per case.

    The graph-level trace says an answer was wrong. It does not say whether the
    evidence was ever in the context — which is the only thing that decides
    whether to work on retrieval or on generation. This span carries the gold
    evidence, what came back, and the per-hop recall, so that question is
    answerable from the trace instead of from a re-run.

    Yields a one-key dict; assign ``d["result"]`` inside the block.
    """
    box: dict[str, Any] = {}
    if langfuse is None:
        yield box
        return
    from langfuse import propagate_attributes

    with (
        langfuse.start_as_current_observation(name="recall", as_type="retriever"),
        propagate_attributes(session_id=session, tags=["locomo"]),
    ):
        yield box
        r = box.get("result") or {}
        langfuse.update_current_span(
            input={"question": question.get("question", ""), "category": question.get("category")},
            output={
                "evidence_recall": r.get("evidence_recall"),
                "hits": r.get("n_hits"),
                "chars": r.get("chars"),
                "gold_evidence": r.get("gold_evidence"),
                "memories": r.get("memories"),
            },
        )
        if (score := r.get("evidence_recall")) is not None and score >= 0:
            # A score, not just metadata: Langfuse can then filter/aggregate the
            # run by it, which is the whole point of recording it per case.
            langfuse.score_current_span(name="evidence_recall", value=float(score))


__all__ = ["retrieval_span", "tracer"]
