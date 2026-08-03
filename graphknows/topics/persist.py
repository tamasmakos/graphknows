"""Writing a TopicSet to the graph, and exporting it as a JSON artifact.

Separate from the builder so a rebuild can be inspected before it replaces what
is in the database, and so a sweep can compute many builds while persisting one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from graphknows.topics.model import TopicSet

log = logging.getLogger(__name__)


async def persist_topics(store: Any, topic_set: TopicSet, *, replace: bool = True) -> int:
    """Write the topics as TOPIC vertices + IN_TOPIC membership edges. Returns the count.

    ``replace`` drops the existing topic layer first, which is the default
    because a rebuild at new settings supersedes the old partition — leaving
    both would let a retrieval channel match topics from two different builds.
    Entities, chunks and relations are untouched either way.
    """
    if replace:
        await _clear(store)
    if not topic_set.topics:
        return 0
    modularity = topic_set.metrics.get("newman_modularity")
    written = await store.write_topics(
        [
            {
                "topic_id": t.topic_id,
                "summary": t.summary,
                "modularity": modularity,
                "title": t.title,
                "keywords": t.keywords,
                "coherence": t.coherence,
                "entities": t.entities,
            }
            for t in topic_set.topics
        ]
    )
    log.info("Topic layer persisted: %d topics", written)
    return int(written)


async def _clear(store: Any) -> None:
    """Remove the previous topic layer (TOPIC vertices take their edges with them)."""
    try:
        await store.command("MATCH (t:TOPIC) DETACH DELETE t")
    except Exception as exc:
        log.debug("topic layer clear skipped (%s)", exc)


def export_json(topic_set: TopicSet, path: str | Path, *, include_embedding: bool = False) -> Path:
    """Write the build to ``path`` as a read-only artifact. Returns the path.

    Read-only by intent: there is no import path. The graph is authoritative, so
    a hand-edited file can never silently overwrite a computed state; re-running
    the stage is how you change what is stored.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = topic_set.to_dict(include_embedding=include_embedding)
    payload["_note"] = (
        "Read-only artifact. Re-run the topic stage to change what is stored; "
        "this file is never imported."
    )
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Topic artifact written: %s (%d topics)", target, len(topic_set))
    return target


__all__ = ["export_json", "persist_topics"]
