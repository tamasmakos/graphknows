"""SC-003/SC-005: an assisted-mode ingest loads none of the local models.

The claim is about a *process*, so it is measured in one. A fresh interpreter
drives a real ingest through ``LLMDecoder`` against a stubbed ``dspy.LM`` and
reports which of the local extraction modules ended up in ``sys.modules``.
Asserting that from inside pytest would prove nothing: the suite has already
imported the local stack for its own tests.

The two missing attributes are the structural half of the same claim (FR-022):
no chunk can fall back to a local decoder on a decoder that has no ``gliner``
and no ``relation_verifier`` to fall back to.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from graphknows.ingestion.extraction.llm.decoder import LLMDecoder

pytestmark = pytest.mark.unit

#: The local extraction stack: the two entity models and the relation-verifier
#: checkpoint module. None of them may load in an assisted-mode process.
LOCAL_MODELS = (
    "gliner",
    "gliner2",
    "graphknows.ingestion.extraction.relations._gliner2_verifier",
)

#: Passed the watched module names as argv, so it needs no escaping here.
_DRIVER = """
import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import dspy

from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.parsers.text import ParsedFile, TextSegment
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.settings import GraphKnowsSettings, MemoryMode

REPLY = json.dumps(
    {
        "entities": [
            {"surface": "Melanie", "label": "person"},
            {"surface": "Acme", "label": "organization"},
        ],
        "relations": [
            {
                "head": "Melanie",
                "predicate": "worksFor",
                "tail": "Acme",
                "evidence": "Melanie works for Acme",
                "confidence": 0.9,
            }
        ],
        "frames": [],
    }
)


class _StubProvider(dspy.LM):
    "The provider, stubbed: no network, one canned structured answer."

    def __init__(self):
        super().__init__(model="openrouter/stub/decoder-v1", api_key="stub", cache=False)

    def __call__(self, prompt=None, messages=None, **kwargs):
        return [REPLY]


def _store():
    store = MagicMock()
    for name in (
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_instance_of",
        "write_temporal",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    return store


decoder = LLMDecoder(
    GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted), lm=_StubProvider()
)
handler = _STMIngestHandler(_store(), extractor=decoder, embed_fn=lambda _text: [0.1])
parsed = ParsedFile(
    source_id="f1",
    title="t",
    parser_name="text",
    segments=[TextSegment(text="Melanie works for Acme.", heading_path="", char_offset=0)],
)
asyncio.run(handler.ingest_parsed(parsed, session_id="s1"))

print("RESULT" + json.dumps([name for name in sys.argv[1:] if name in sys.modules]))
"""


def _loaded_after_an_assisted_ingest() -> list[str]:
    """Run the ingest in a fresh process; report the local models it loaded."""
    pytest.importorskip("dspy")  # the provider stub subclasses dspy.LM
    out = subprocess.run(
        [sys.executable, "-c", _DRIVER, *LOCAL_MODELS],
        capture_output=True,
        text=True,
        check=True,
    )
    result = [line for line in out.stdout.splitlines() if line.startswith("RESULT")]
    assert result, f"the ingest printed no result\n{out.stdout}\n{out.stderr}"
    return list(json.loads(result[-1].removeprefix("RESULT")))


def test_an_assisted_ingest_loads_no_local_extraction_model() -> None:
    assert _loaded_after_an_assisted_ingest() == []


def test_the_decoder_has_no_local_model_to_fall_back_to() -> None:
    # On the class, so the claim holds before any instance is built.
    assert not hasattr(LLMDecoder, "gliner")
    assert not hasattr(LLMDecoder, "relation_verifier")
