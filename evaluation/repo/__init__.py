"""Repository-transcript panel row: this repository's own agent work as a benchmark.

Questions are answered from the transcripts of agents working in this repository,
with gold answers taken from git history and the transcripts themselves (decision
location, change rationale, which test guards a symbol). Same pipeline, same
metric shape, same DSPy judge as every other row.
"""

from evaluation.repo.adapter import RepoAdapter
from evaluation.repo.config import RepoConfig
from evaluation.repo.dataset import load_units

__all__ = ["RepoAdapter", "RepoConfig", "load_units"]
