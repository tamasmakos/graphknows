"""LoCoMo benchmark: long-term conversational memory QA (arXiv 2402.17753).

10 multi-session dialogues between two speakers; questions span single-hop,
multi-hop (knowledge synthesis), temporal, and open-ended categories.
"""

from evaluation.locomo.adapter import LocomoAdapter
from evaluation.locomo.config import LocomoConfig
from evaluation.locomo.dataset import load_units

__all__ = ["LocomoAdapter", "LocomoConfig", "load_units"]
