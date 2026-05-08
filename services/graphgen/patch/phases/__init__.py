"""Ontology patch phases."""

from .legacy import run_legacy_phase
from .structural import run_structural_phase
from .unknown import run_unknown_phase

__all__ = [
    "run_structural_phase",
    "run_legacy_phase",
    "run_unknown_phase",
]
