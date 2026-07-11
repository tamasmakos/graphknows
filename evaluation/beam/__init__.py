"""BEAM benchmark: billion-scale episodic and agentic memory (ICLR 2026).

100 synthetic long conversations per size bucket (100K → 10M tokens), each with
20 probing questions across 10 memory-ability types (abstention, contradiction
resolution, event ordering, information extraction, instruction following,
knowledge update, multi-session reasoning, preference following, summarization,
temporal reasoning). Answers are graded against per-question rubric nuggets.
"""

from evaluation.beam.adapter import BeamAdapter
from evaluation.beam.config import BeamConfig
from evaluation.beam.dataset import load_units

__all__ = ["BeamAdapter", "BeamConfig", "load_units"]
