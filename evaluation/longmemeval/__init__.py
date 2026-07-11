"""LongMemEval benchmark: long-term interactive memory for chat assistants
(arXiv 2410.10813, ICLR 2025).

500 questions, each carrying its own multi-session user↔assistant haystack.
Six question types: single-session-user, single-session-assistant,
single-session-preference, multi-session, temporal-reasoning, knowledge-update.
Questions whose id ends in ``_abs`` are abstention probes.
"""

from evaluation.longmemeval.adapter import LongMemEvalAdapter
from evaluation.longmemeval.config import LongMemEvalConfig
from evaluation.longmemeval.dataset import load_units

__all__ = ["LongMemEvalAdapter", "LongMemEvalConfig", "load_units"]
