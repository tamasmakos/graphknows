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

    Both are settings-gated. Frames default ON (``enable_frames``); ontology
    defaults OFF (``enable_ontology_channel``) because it has never been
    measured on its own. It used to be gated on ``settings.ontology_source``,
    which cannot be false — that property always resolves to an ontology — so
    making injection always-on silently put a sixth leg into the RRF fusion. The
    next full conv-30 run fused ontology 838 alongside semantic 1875, text 1588,
    entity 1498, frame 1312, temporal 625 and scored accuracy 0.766 -> 0.722
    (judge noise ±0.02) with retrieval misses 8 -> 13. That run carried
    extraction changes as well, so the channel is a suspect, not a convict: this
    restores the five-channel set the 0.766 was measured on so the next A/B
    varies one thing. See ``GraphKnowsSettings.enable_ontology_channel``.
    """
    channels: list[Channel] = []
    if settings.enable_frames:
        from graphknows.channels.frames import FrameChannel

        channels.append(FrameChannel())
    if settings.enable_frame_facts:
        from graphknows.channels.frame_facts import FrameFactChannel

        channels.append(FrameFactChannel())
    if settings.enable_ontology_channel:
        from graphknows.channels.ontology import OntologyChannel

        channels.append(OntologyChannel())
    # Unconditional, unlike the two above: WordNetChannel only populates (a
    # flush-time backfill) and never implements collect, so it contributes no
    # leg to the RRF fusion and cannot perturb retrieval the way a settings
    # gate protects against for enable_ontology_channel.
    from graphknows.channels.wordnet import WordNetChannel

    channels.append(WordNetChannel())
    return channels


__all__ = ["default_channels"]
