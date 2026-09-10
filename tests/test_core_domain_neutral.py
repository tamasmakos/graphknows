"""SC-001: no benchmark, dialogue, speaker or turn-regex term in core code.

`.importlinter` stops the core *importing* a pack; this stops the core
*naming* one domain. Only real code is scanned — a docstring may still say
"the dialogue pack" — so the check is over Python tokens, not raw text.

The old core still carries the dialogue vocabulary until the cutover (T037,
FR-045), so those files sit in a quarantine list that only shrinks: a new
offender fails the first test, and a cleaned-up file fails the second until
it is struck off.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "graphknows"

DOMAIN_TERM = re.compile(
    r"benchmark|dialogue|speaker|utterance|locomo|longmem|turn_(id|index|span|regex)",
    re.IGNORECASE,
)

# Dialogue-shaped modules awaiting the cutover (T037). Strike each off there.
PENDING_CUTOVER = frozenset(
    {
        "ingestion/extraction/entities/extractor.py",
        "ingestion/extraction/llm/anchor.py",
        "ingestion/extraction/llm/decoder.py",
        "ingestion/extraction/llm/schema.py",
        "ingestion/extraction/protocol.py",
        "ingestion/extraction/relations/_lingfeatures.py",
        "ingestion/parsers/text.py",
        "integrations/langgraph/_session.py",
        "memory.py",
        "models/hit.py",
        "models/ingest_options.py",
    }
)


def domain_terms(module: Path) -> list[str]:
    """Domain-specific names used as code in `module`, comments and strings aside."""
    tokens = tokenize.generate_tokens(io.StringIO(module.read_text(encoding="utf-8")).readline)
    return [
        token.string
        for token in tokens
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
        and DOMAIN_TERM.search(token.string)
    ]


def core_modules() -> list[Path]:
    return sorted(path for path in CORE.rglob("*.py") if "packs/" not in path.as_posix())


@pytest.mark.parametrize("module", core_modules(), ids=lambda path: path.name)
def test_core_module_names_no_domain(module: Path) -> None:
    relative = module.relative_to(CORE).as_posix()
    if relative in PENDING_CUTOVER:
        pytest.skip("dialogue-shaped until the cutover (T037)")
    assert not domain_terms(module), f"{relative} names a domain the core must not know"


def test_quarantine_holds_only_modules_that_still_offend() -> None:
    clean = {name for name in PENDING_CUTOVER if not domain_terms(CORE / name)}
    assert not clean, f"cleaned up — drop from PENDING_CUTOVER: {sorted(clean)}"
