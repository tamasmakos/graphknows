"""LongMemEval benchmark: long-term memory QA over user/assistant chat history.

500 questions across six categories (single-session user/assistant/preference
recall, knowledge update, temporal reasoning, multi-session). Each question
carries its own ~53-session haystack; the evidence turns are flagged inline.
Judged with the shared DSPy SemanticF1 judge (``evaluation.common.dspy_judge``).
"""

from evaluation.longmem.adapter import LongMemAdapter
from evaluation.longmem.config import LongMemConfig
from evaluation.longmem.dataset import load_units

__all__ = ["LongMemAdapter", "LongMemConfig", "load_units"]
