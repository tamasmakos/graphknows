"""Topic modeling — a re-runnable stage over an already-ingested graph.

Topics are the one layer you want to rebuild repeatedly at different settings,
so this is a stage rather than a step inside ``flush_memory``: rebuilding must
not re-run entity resolution, PageRank, node2vec and consolidation, let alone a
re-ingest.

The pipeline is ``build_topics`` (read + compute, no writes) →
``persist_topics`` / ``export_json``. Splitting compute from write is what lets
a sweep evaluate many parameter sets and store only the winner.

Two facts about the underlying engine shaped this design, both verified against
ArcadeDB 26.5.1:

- ``algo.louvain`` **ignores its parameter map** — resolution 0.1 and 5.0 return
  identical partitions — so the community layer offers no granularity knob;
  ``min_size`` is the only lever. A node2vec-clustering source once traded that
  away for a real knob but was removed for coverage — see
  :mod:`graphknows.topics.config`.
- ``algo.modularityScore`` is **not** Newman modularity and must not be used as
  a tuning objective: an all-in-one-cluster partition scores 0.750 against a
  real Louvain partition's 0.247. See :mod:`graphknows.topics.metrics`.
"""

from graphknows.topics.builder import build_topics
from graphknows.topics.config import TopicConfig
from graphknows.topics.metrics import newman_modularity
from graphknows.topics.model import Topic, TopicSet
from graphknows.topics.persist import export_json, persist_topics

__all__ = [
    "Topic",
    "TopicConfig",
    "TopicSet",
    "build_topics",
    "export_json",
    "newman_modularity",
    "persist_topics",
]
