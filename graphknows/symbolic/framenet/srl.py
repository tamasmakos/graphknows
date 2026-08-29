"""Transformer frame identification: FrameNet triggers + FE spans, per sentence.

The alternative to :func:`~graphknows.ingestion.extraction.relations.frame_srl.lu_triggers`,
which *guesses* a chunk's frames from its clause main verbs, nominalisations and
light-verb constructions ranked by definition-embedding fit.

This parses instead: a seq2seq FrameNet parser (frame-semantic-transformer)
returns every frame-evoking trigger in a sentence with its own frame and frame
element spans, so each fact gets its own instance. Role FILLING still belongs to
the GLiNER2 core-FE schema downstream — the parser supplies the frame and the
trigger, not the entity names.

Optional: needs the ``srl`` extra. Nothing imports this module unless
``GRAPHKNOWS_FRAME_SRL`` is on.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from graphknows.exceptions import MissingExtraError

_model: Any = None
# Sentence text -> its parse, with SENTENCE-LOCAL offsets (see _parse_sentences).
_sentence_cache: dict[str, tuple[tuple[str, int, tuple], ...]] = {}
_CACHE_MAX = 20_000

# One sentence per parse: the parser is sentence-level, and a chat turn without
# final punctuation must still split from the next speaker's line.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\w+")


@lru_cache(maxsize=1)
def installed() -> bool:
    """Whether the ``srl`` extra is importable, without importing it."""
    return importlib.util.find_spec("frame_semantic_transformer") is not None


def srl_enabled() -> bool:
    """Whether transformer frame identification replaces the LU/embedding guess.

    Defaults to ON **when the extra is installed** — measured on conv-30 (81q,
    turn-fed): accuracy 0.778 -> 0.889, evidence_recall_in_context 0.765 ->
    0.926, retrieval-bound failures 13 -> 3. A base install has no parser, so it
    keeps the LU/embedding identifier and nothing breaks.

    ``GRAPHKNOWS_FRAME_SRL`` overrides either way and is read at call time, so an
    eval can still flip it per subprocess. It buys accuracy with ingest time: a
    seq2seq call per sentence and again per detected trigger.
    """
    override = os.environ.get("GRAPHKNOWS_FRAME_SRL", "").strip().lower()
    if override:
        return override in ("1", "true", "yes", "fst")
    return installed()


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
            # The library batches 8 sentences per generate call, which leaves a
            # GPU mostly idle. Ingest hands us a whole chunk at once, so a bigger
            # batch is free throughput; lower it if VRAM is tight (6 GB fits 32).
            batch_size=int(os.environ.get("GRAPHKNOWS_FRAME_SRL_BATCH", "32")),
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


_SPEAKER_PREFIX = re.compile(r"^\w+:\s+")


def marked_line(text: str, offset: int, word: str) -> str:
    """The SENTENCE containing *offset*, with the trigger at *offset* starred.

    Sentence-scoped, not line-scoped (issue #161): in a chat chunk a line is a
    whole turn and can state several events ("Tom got fired... Caroline got
    hired..."), so a line-wide role-filling context let one event's roles pull
    in the other's entities (a Firing and a Hiring instance sharing all four
    fillers). Reusing :func:`sentences` keeps a sentence from ever crossing a
    line, so this can only narrow, never widen, what the old line-based version
    returned.

    When the trigger's sentence is not the line's first, the line's
    ``Speaker:`` prefix — detected from the start of the line, not the
    sentence — is re-prepended: it carries no text of the event itself, but is
    load-bearing for resolving a 1st/2nd-person filler ("I") to the speaker.

    The star still separates two triggers of one frame inside one sentence
    ("I gave a book and gave a pen") — naming the trigger word in the role
    schema cannot, since both are 'gave'. Marking it in the TEXT gives each
    instance its own input, and therefore its own roles.
    """
    line_start = text.rfind("\n", 0, offset) + 1
    sent_start, sent = next(s for s in sentences(text) if s[0] <= offset < s[0] + len(s[1]))
    local = offset - sent_start
    marked = f"{sent[:local]}*{sent[local : local + len(word)]}*{sent[local + len(word) :]}"
    if sent_start == line_start:
        return marked
    line_end = text.find("\n", line_start)
    line = text[line_start : len(text) if line_end < 0 else line_end]
    prefix = m.group(0) if (m := _SPEAKER_PREFIX.match(line)) else ""
    return f"{prefix}{marked}"


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


def _parse_sentences(sents: list[str]) -> dict[str, tuple[tuple[str, int, tuple], ...]]:
    """Parse the sentences that are not cached; return every one's SENTENCE-LOCAL hits.

    Cached because a stored CHUNK is a 5-turn sliding window at stride 3, so
    each turn belongs to ~1.7 chunks and would otherwise be parsed that many
    times — ~40% of the parser's work on a conversation is re-parsing text it
    has already seen. Offsets are kept sentence-local so a cached parse can be
    rebased onto whichever chunk asks for it next.
    """
    todo = [s for s in dict.fromkeys(sents) if s not in _sentence_cache]
    if todo:
        for sent, result in zip(todo, _load().detect_frames_bulk(todo), strict=True):
            local: list[tuple[str, int, tuple]] = []
            for fr in result.frames:
                fes = tuple(
                    (fe.name, i, i + len(fe.text))
                    for fe in fr.frame_elements
                    # First occurrence: FE spans rarely repeat verbatim in one sentence.
                    if (i := sent.find(fe.text)) >= 0
                )
                local.append((fr.name, fr.trigger_location, fes))
            if len(_sentence_cache) >= _CACHE_MAX:
                _sentence_cache.clear()  # ponytail: whole-cache drop, LRU if it ever churns
            _sentence_cache[sent] = tuple(local)
    return _sentence_cache


def prewarm(texts: list[str]) -> int:
    """Parse every sentence of *texts* in one batched pass; return how many were new.

    Ingest knows all of a session's chunks before it writes any of them, so the
    parser can see hundreds of sentences at once instead of ~14 per chunk — the
    same batching the entity pass already does. After this, each chunk's
    :func:`parse` is a cache hit.
    """
    sents = [s for t in texts for _, s in sentences(t)]
    before = len(_sentence_cache)
    if sents:
        _parse_sentences(sents)
    return len(_sentence_cache) - before


def parse(text: str) -> list[FrameHit]:
    """Every frame-evoking trigger in *text*, with chunk-global offsets."""
    hits: list[FrameHit] = []
    pairs = sentences(text)
    if not pairs:
        return hits
    cache = _parse_sentences([s for _, s in pairs])
    for offset, sent in pairs:
        for frame, trigger_at, fes in cache.get(sent, ()):
            at = offset + trigger_at
            word = m.group(0) if (m := _WORD.match(text[at:])) else ""
            hits.append(
                FrameHit(
                    frame=frame,
                    trigger=word,
                    offset=at,
                    fes=tuple((role, offset + s, offset + e) for role, s, e in fes),
                )
            )
    return hits
