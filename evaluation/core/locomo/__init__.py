"""LoCoMo benchmark adapter."""

from evaluation.core.file_backed import FileBackedBenchmarkOptions
from evaluation.core.locomo.runner import run_locomo_benchmark

__all__ = ["FileBackedBenchmarkOptions", "run_locomo_benchmark"]
