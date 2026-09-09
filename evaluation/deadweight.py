"""The dead-weight check: written vertex/edge types minus the ones read (FR-040).

A plane that is written on every ingest and never read back is dead weight — cost
with no retrieval to show for it. The check is the set difference, and the result
is a run outcome rather than an advisory line: ``deadweight`` and ``panel
--compare`` call :meth:`DeadWeight.gate` and exit non-zero, while ``baseline``
records :meth:`DeadWeight.to_record` and continues, since the baseline's job is to
write down what the old core does, dead planes included (FR-041).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeadWeight:
    """The types written during a run that nothing read back."""

    unread_types: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.unread_types

    def to_record(self) -> dict[str, Any]:
        """The contract's ``unread_types``/``passed`` shape (contracts/panel-report.md)."""
        return {"unread_types": list(self.unread_types), "passed": self.passed}

    def gate(self) -> None:
        """Fail the run when a written type has no reader; a no-op when it passes."""
        if not self.passed:
            raise SystemExit(f"dead weight: written but never read: {', '.join(self.unread_types)}")


def check(written: Iterable[str], read: Iterable[str]) -> DeadWeight:
    """Diff the types *written* during a run against the ones *read* back."""
    return DeadWeight(tuple(sorted(set(written) - set(read))))
