"""BEAM benchmark: production-scale agent memory evaluation.

Conversations at large token volumes (this is the 100K-token tier) probed across
ten categories: preference following, instruction following, information
extraction, knowledge update, multi-session reasoning, summarization, temporal
reasoning, event ordering, abstention, and contradiction resolution.

Gold-answer categories are graded with the shared DSPy SemanticF1 judge;
instruction/preference following (which have no single gold answer) are graded
against their compliance rubric via the shared rubric judge.
"""

from evaluation.beam.adapter import BeamAdapter
from evaluation.beam.config import BeamConfig
from evaluation.beam.dataset import load_units

__all__ = ["BeamAdapter", "BeamConfig", "load_units"]
