"""``RelationVerifier.from_pretrained`` assembles a usable verifier from a checkpoint.

The loader was reachable only through a real Hugging Face download, so the one
existing test (``test_gate_load_failure_passes_through``) could exercise nothing
past the two ``hf_hub_download`` attempts — every line that assembles the model
from what was downloaded was untested.

What is asserted is the checkpoint contract: the ``.pt``/``.bin`` filename
fallback, and that a plain ``state_dict`` — the shape the upstream checkpoint
ships — loads into ``RelationVerifierModel`` without a wrapper key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from graphknows.ingestion.extraction.relations._gliner2_verifier import (
    RelationVerifier,
    RelationVerifierModel,
    VerifierConfig,
)

pytestmark = pytest.mark.unit


class _StubEncoder:
    """Stands in for the deberta encoder: loaded by name, moved to a device, not run."""

    name: str

    @classmethod
    def from_pretrained(cls, name: str, *_args: Any, **_kwargs: Any) -> _StubEncoder:
        stub = cls()
        stub.name = name
        return stub

    def to(self, _device: str) -> _StubEncoder:
        return self

    def eval(self) -> _StubEncoder:
        return self


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """A checkpoint with exactly the keys ``RelationVerifierModel`` declares."""
    path = tmp_path / "verifier.pt"
    torch.save(RelationVerifierModel(VerifierConfig()).state_dict(), path)
    return path


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the encoder and tokenizer loads; both only need to be named."""
    import transformers

    monkeypatch.setattr(transformers, "AutoModel", _StubEncoder)
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubEncoder)


def test_loads_a_bare_state_dict(
    checkpoint: Path, patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from graphknows.ingestion.extraction.relations import _gliner2_verifier

    monkeypatch.setattr(_gliner2_verifier, "hf_hub_download", lambda **_kw: str(checkpoint))

    verifier = RelationVerifier.from_pretrained("fake/verifier", threshold=0.42)

    assert isinstance(verifier, RelationVerifier)
    assert verifier.config.threshold == 0.42
    assert verifier.encoder.name == VerifierConfig().encoder_name


def test_falls_back_to_the_bin_filename(
    checkpoint: Path, patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream ships either ``verifier.pt`` or ``pytorch_model.bin``."""
    from graphknows.ingestion.extraction.relations import _gliner2_verifier

    asked: list[str] = []

    def _download(*, repo_id: str, filename: str) -> str:
        asked.append(filename)
        if filename == "verifier.pt":
            raise OSError("404")
        return str(checkpoint)

    monkeypatch.setattr(_gliner2_verifier, "hf_hub_download", _download)

    RelationVerifier.from_pretrained("fake/verifier")

    assert asked == ["verifier.pt", "pytorch_model.bin"]
