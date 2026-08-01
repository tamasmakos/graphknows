"""The channels active for a configuration.

Only optional signals are channels. The four baseline collectors (entity,
vector, bm25, temporal) are methods on the retriever spine and are always run;
they used to be wrapped in ``Channel`` subclasses whose entire body was
``return await rt.<method>(...)``, behind a name-keyed registry and a
profile filter that never excluded anything.

No graph-traversal channel. Measured on conv-30, ``ppr`` returned 367 hits for
12 gold-bearing ones (3.3% — the worst of six channels) and was the ONLY
consumer of REL edges, which is why every improvement to the relation layer
measured flat on retrieval. ``topic`` is gone because nothing on the read path
consumed what the topic stage wrote: its only retrieval reader was
``DETRetriever.topic_match``, itself orphaned. (It is *not* gone because
"topic_mode is none in both presets" — that was the stated reason and it is
wrong: the ``llm_assisted`` preset sets ``TopicMode.llm``.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphknows.channels.base import Channel
    from graphknows.settings import GraphKnowsSettings


def default_channels(settings: GraphKnowsSettings) -> list[Channel]:
    """Instantiate the optional channels this configuration turns on.

    Frames are opt-in via ``settings.enable_frames``. Ontology is data-gated:
    without an ontology there are no ONTOLOGY_CLASS nodes to search, so the
    channel would cost a vector query per call to return nothing.
    """
    channels: list[Channel] = []
    if settings.enable_frames:
        from graphknows.channels.frames import FrameChannel

        channels.append(FrameChannel())
    if settings.ontology_source:
        from graphknows.channels.ontology import OntologyChannel

        channels.append(OntologyChannel())
    return channels


__all__ = ["default_channels"]
