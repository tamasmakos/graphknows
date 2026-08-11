"""Transformer frame identification: FrameNet triggers + FE spans, per sentence.

The alternative to :func:`~graphknows.ingestion.extraction.relations.frame_srl.lu_candidate_frames`,
which *guesses* a chunk's frames from its clause main verbs ranked by
definition-embedding fit. That guess is per CHUNK, so the role-fillability gate
downstream keeps at most one instance for a five-turn window — a chunk stating
five facts contributes one.

This parses instead: a seq2seq FrameNet parser (frame-semantic-transformer)
returns every frame-evoking trigger in a sentence with its own frame and frame
element spans, so each fact gets its own instance. Role FILLING still belongs to
the GLiNER2 core-FE schema downstream — the parser supplies the frame and the
trigger, not the entity names.

Optional: needs the ``srl`` extra. Nothing imports this module unless
``GRAPHKNOWS_FRAME_SRL`` is on.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from graphknows.exceptions import MissingExtraError

_model: Any = None

# One sentence per parse: the parser is sentence-level, and a chat turn without
# final punctuation must still split from the next speaker's line.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\w+")


def srl_enabled() -> bool:
    """Whether transformer frame identification replaces the LU/embedding guess.

    Read at call time (not import time) so an eval A/B can flip
    ``GRAPHKNOWS_FRAME_SRL`` per subprocess, as the other ingest knobs do.
    Default OFF: the parser is a seq2seq model run once per sentence and again
    per detected trigger, which is a large ingest cost to take by default.
    """
    return os.environ.get("GRAPHKNOWS_FRAME_SRL", "").strip().lower() in ("1", "true", "yes", "fst")


@dataclass(frozen=True)
class FrameHit:
    """One frame-evoking trigger, with offsets into the chunk text."""

    frame: str
    trigger: str
    offset: int
    fes: tuple[tuple[str, int, int], ...] = ()


def _load() -> Any:
    """The parser, loaded once per process."""
    global _model
    if _model is None:
        try:
            from frame_semantic_transformer import (  # type: ignore[import-untyped]
                FrameSemanticTransformer,
            )
        except ImportError as exc:  # pragma: no cover - exercised by the extras test
            raise MissingExtraError("Transformer frame identification", "srl") from exc

        # predictions_per_sample IS num_beams. Do not lower it to 1 for speed:
        # the parser generates with repetition_penalty=2.5 and transformers
        # ignores early_stopping outside beam search, so a single sequence runs
        # to max_length=512 every call while 5 beams stop at EOS in a few dozen
        # steps. Greedy measured several times SLOWER end to end.
        _model = FrameSemanticTransformer(
            os.environ.get("GRAPHKNOWS_FRAME_SRL_MODEL", "base"),
            predictions_per_sample=int(os.environ.get("GRAPHKNOWS_FRAME_SRL_BEAMS", "5")),
            use_gpu=_use_gpu(),
        )
    return _model


def _use_gpu() -> bool:
    """GPU unless ``GRAPHKNOWS_FRAME_SRL_DEVICE=cpu`` (mirrors the relex knob).

    Worth forcing off when the GPU is shared: this parser queues behind whatever
    else holds it, and a contended GPU is far slower than the CPU path.
    """
    if os.environ.get("GRAPHKNOWS_FRAME_SRL_DEVICE", "").strip().lower() == "cpu":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def sentences(text: str) -> list[tuple[int, str]]:
    """``(offset, sentence)`` pairs, offsets into *text*."""
    out: list[tuple[int, str]] = []
    pos = 0
    for part in _SPLIT.split(text.strip()):
        if not part:
            continue
        start = text.index(part, pos)
        out.append((start, part))
        pos = start + len(part)
    return out


def marked_line(text: str, offset: int, word: str) -> str:
    """The line containing *offset*, with the trigger at *offset* starred.

    The LINE, not the sentence: in a chat chunk the line is the turn, so it
    keeps the ``Speaker:`` prefix that lets the role filler resolve "I".

    The star is what separates two instances of one frame on one line ("I gave a
    book and gave a pen") — naming the trigger word in the role schema cannot,
    since both are 'gave'. Marking it in the TEXT gives each instance its own
    input, and therefore its own roles.
    """
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    line, local = text[start : len(text) if end < 0 else end], offset - start
    return f"{line[:local]}*{line[local : local + len(word)]}*{line[local + len(word) :]}"


def nested_pairs(hits: list[FrameHit]) -> list[tuple[int, str, int]]:
    """``(parent_index, role, child_index)`` where a role span embeds another trigger.

    The structure a flat role list cannot hold: "I bought it because my laptop
    died" fills Commerce_buy's *Explanation* with a span that is itself a
    Causation frame, whose *Cause* is a Death frame. Persisted, that is a
    traversable chain instead of one long string filler.

    Only the OUTERMOST trigger inside a span attaches. Without that filter a wide
    span claims every frame it covers (Explanation -> Causation AND Age AND
    Death), which is three edges asserting the same nesting at three depths; the
    one that is not itself inside a sibling's role span is the head.
    """
    spans_of = {i: h.fes for i, h in enumerate(hits)}
    out: list[tuple[int, str, int]] = []
    for parent, hit in enumerate(hits):
        for role, start, end in hit.fes:
            inside = [i for i, h in enumerate(hits) if i != parent and start <= h.offset < end]
            for child in inside:
                covered_by_sibling = any(
                    s <= hits[child].offset < e
                    for other in inside
                    if other != child
                    for _, s, e in spans_of[other]
                )
                if not covered_by_sibling:
                    out.append((parent, role, child))
    return out


def parse(text: str) -> list[FrameHit]:
    """Every frame-evoking trigger in *text*, with chunk-global offsets."""
    hits: list[FrameHit] = []
    pairs = sentences(text)
    if not pairs:
        return hits
    sents = [s for _, s in pairs]
    parsed = _load().detect_frames_bulk(sents)
    for (offset, sent), result in zip(pairs, parsed, strict=True):
        for fr in result.frames:
            fes = []
            for fe in fr.frame_elements:
                # First occurrence: FE spans rarely repeat verbatim in one sentence.
                i = sent.find(fe.text)
                if i >= 0:
                    fes.append((fe.name, offset + i, offset + i + len(fe.text)))
            at = offset + fr.trigger_location
            word = m.group(0) if (m := _WORD.match(text[at:])) else ""
            hits.append(FrameHit(frame=fr.name, trigger=word, offset=at, fes=tuple(fes)))
    return hits
