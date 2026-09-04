"""``LLMDecoder`` — one structured call per chunk, in place of the local stack.

The seam is duck-typed (contracts/decoder.md §1): this class answers
``extract``/``extract_batch``/``nlp``/``parse`` exactly as
``GLiNER2EntityExtractor`` does, and deliberately answers **nothing** to
``gliner`` or ``relation_verifier``. Absence is the capability signal the
consumers already read with ``hasattr``/``getattr``, and it is what makes
"loads zero local extraction models" structural rather than a promise.

Two wiring decisions carry the strict-schema guarantee (research.md R4):

* the configured model is registered with litellm as schema-capable at
  construction — litellm's model map has no entry for the shipped default, and
  without one a strict ``json_schema`` request is downgraded to
  ``{"type": "json_object"}``, so the schema never reaches the provider;
* the call goes through a decoder-owned ``dspy.JSONAdapter``, never the ambient
  adapter — ``ChatAdapter`` sends no ``response_format`` at all, and prompts for
  its own bracketed sections instead.

The adapter's ``format`` and ``parse`` are driven directly rather than through
its ``__call__``, which re-asks the provider in plain JSON mode whenever an
answer does not parse. One chunk is one request (FR-010) — on the failure path
too, where a silent re-ask would be invisible in the report.

``extract_batch`` fans the per-chunk calls out over a bounded pool
(``GRAPHKNOWS_DECODE_CONCURRENCY``). There is no local model to serialise on
here — the whole per-chunk cost is network wait, ~26 s of it measured on
conv-30 — so the bound is the only thing that has to hold.
"""

from __future__ import annotations

import functools
import logging
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from graphknows.ingestion.extraction.entities.extractor import (
    _BASE_ENTITY_LABELS,
    ExtractionResult,
)
from graphknows.ingestion.extraction.llm.anchor import ResponseGates
from graphknows.ingestion.extraction.llm.schema import (
    DecodedEntity,
    DecodedFrameInstance,
    DecodedRelation,
    DecodeRequest,
    DecodeResponse,
    FrameCandidate,
)
from graphknows.llm import build_lm
from graphknows.nlp import load_spacy_model

log = logging.getLogger(__name__)


def _dspy() -> Any:
    """dspy, imported on first use rather than at module import.

    Importing dspy imports litellm; neither belongs in a process that only ever
    runs the local decoder, and the base install does not carry them at all.
    """
    try:
        import dspy
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        from graphknows.exceptions import MissingExtraError

        raise MissingExtraError("llm decoding", "assisted") from exc
    return dspy


@functools.cache
def _decode_signature() -> Any:
    """The full ``Decode`` signature; sections are trimmed per chunk by :func:`_signature_for`."""
    dspy = _dspy()

    class Decode(dspy.Signature):  # type: ignore[misc,name-defined]
        """Label one chunk against the closed vocabularies offered with it.

        Copy every surface VERBATIM from the text — a surface that is not a literal
        substring of it is discarded. Use only the labels, predicates and role names
        on offer; a near miss is discarded too, never mapped onto a neighbour.
        """

        text: str = dspy.InputField(desc="The chunk to label.")
        speaker: str = dspy.InputField(desc="Who uttered the chunk; empty when unknown.")
        entity_labels: list[str] = dspy.InputField(desc="The only entity labels on offer.")
        relation_spec: dict[str, str] = dspy.InputField(
            desc="The only predicates on offer, as label -> definition."
        )
        frame_candidates: list[dict[str, Any]] = dspy.InputField(
            desc=(
                "Frames whose roles are to be filled, as {frame, trigger, core_elements}. "
                "Fill roles for these only; never propose a frame of your own."
            )
        )
        entities: list[dict[str, Any]] = dspy.OutputField(desc="The entities found in the text.")
        relations: list[dict[str, Any]] = dspy.OutputField(
            desc="The relations found in the text, each with a confidence in [0, 1]."
        )
        frames: list[dict[str, Any]] = dspy.OutputField(
            desc="One entry per candidate frame the text fills roles for."
        )

    return Decode


class _StrictSchema(BaseModel):
    """Base of the per-chunk ``response_format`` model: unknown keys are violations."""

    model_config = ConfigDict(extra="forbid")


#: What each response section is asked for with, and the strict item shape it is
#: asked for in. A section whose input is empty is dropped from the signature
#: whole, so it is neither prompted for nor in the schema (contracts/decoder.md §2).
_SECTIONS: tuple[tuple[str, str, Any], ...] = (
    ("entity_labels", "entities", list[DecodedEntity]),
    ("relation_spec", "relations", list[DecodedRelation]),
    ("frame_candidates", "frames", list[DecodedFrameInstance]),
)


def _signature_for(request: DecodeRequest) -> Any:
    """``Decode`` minus every section this chunk has no input for."""
    signature = _decode_signature()
    for source, section, _ in _SECTIONS:
        if not getattr(request, source):
            log.debug("decode section omitted: %s is empty, so no %s asked for", source, section)
            signature = signature.delete(source).delete(section)
    return signature


def _response_format(signature: Any) -> type[BaseModel]:
    """The strict item shape handed to the provider as ``response_format``.

    Built from the item models, not from the container the reply is parsed back
    with: the provider is held to the strict shape, while a single malformed
    item costs itself rather than the chunk (contracts/decoder.md §3).
    """
    fields: dict[str, Any] = {
        section: (items, ...)
        for _, section, items in _SECTIONS
        if section in signature.output_fields
    }
    return create_model("DecodeResponseSchema", __base__=_StrictSchema, **fields)


def _inputs_for(request: DecodeRequest, signature: Any) -> dict[str, Any]:
    """The request, restricted to the fields the trimmed signature still asks for."""
    offered = {
        "text": request.text,
        "speaker": request.speaker,
        "entity_labels": list(request.entity_labels),
        "relation_spec": request.relation_spec,
        "frame_candidates": [asdict(candidate) for candidate in request.frame_candidates],
    }
    return {name: value for name, value in offered.items() if name in signature.input_fields}


def _completion_text(completion: Any) -> str:
    """The text of one completion, however this dspy version boxes it.

    An empty completion is what a truncated reply looks like once the budget
    went on reasoning: dspy hands back ``None`` for the content. Named here,
    because indexing into it raised a ``TypeError`` that read as a bug in the
    decoder rather than as the provider returning nothing.
    """
    text = completion["text"] if isinstance(completion, dict) else completion
    if not text:
        raise ValueError("empty completion: the provider returned no content (truncated?)")
    return text


def _entity_labels(extra: Sequence[str] | None) -> tuple[str, ...]:
    """The base inventory unioned with this chunk's ontology classes, order kept.

    The local path unions the same base set with its spaCy observation; mining
    that costs a parse and buys a model that reads the chunk itself nothing.
    """
    return tuple(dict.fromkeys((*_BASE_ENTITY_LABELS, *(extra or ()))))


def _openrouter_request_options(model: str) -> dict[str, Any]:
    """Per-request routing for OpenRouter models; empty for any other endpoint.

    Two fields, both measured on the shipped model with a real chunk request
    (cache off, `json_schema` on):

    * ``reasoning.enabled = false`` — labelling a chunk against a closed
      vocabulary is not a reasoning task, and the reasoning is what blew the
      budget: 2.4k-4.1k reasoning tokens per call, one reply truncated to no
      content at 4096. Off, the reply is ~300 tokens.
    * ``provider.sort = "throughput"`` — the endpoints that honour a strict
      schema differ 10x in speed (7 vs 93 tok/s on consecutive calls). Sorted,
      the same request took 3.8 s twice; unsorted, 3 s then 54 s.

    Together: ~60 s -> ~4 s per chunk. Both are OpenRouter's own request
    fields, so they are sent only when the model is routed through it.
    """
    if not model.startswith("openrouter/"):
        return {}
    return {"extra_body": {"reasoning": {"enabled": False}, "provider": {"sort": "throughput"}}}


def _register_response_schema(model: str) -> None:
    """Declare *model* schema-capable to litellm (research.md R4).

    Unregistered, the shipped default is unknown to litellm's model map, and a
    strict ``json_schema`` request for it is downgraded to
    ``{"type": "json_object"}``. One documented line is the whole fix; no
    capability probe, no allow-list.
    """
    import litellm

    litellm.register_model({model: {"supports_response_schema": True}})


class LLMDecoder:
    """A decoder that labels a chunk with one structured provider call.

    Neither ``gliner`` nor ``relation_verifier`` exists on it, and that is the
    point: the local fallback is structurally unreachable rather than merely
    unused.
    """

    def __init__(self, settings: Any | None = None, *, lm: Any | None = None) -> None:
        from graphknows.settings import GraphKnowsSettings

        self._settings = settings if settings is not None else GraphKnowsSettings()
        self._lm = lm if lm is not None else self._build_lm()
        _register_response_schema(self._lm.model)
        # Pinned, never the ambient adapter: this decoder's call is the only one
        # whose schema wiring is guaranteed.
        self._adapter = _dspy().JSONAdapter()
        # One-entry memo behind `parse()`, as on the local decoder.
        self._doc_cache: tuple[str, Any] | None = None
        self._abstentions: Counter[str] = Counter()

    def __repr__(self) -> str:
        return f"LLMDecoder(model={self._lm.model!r})"

    def _build_lm(self) -> Any:
        """The decoder's own LM. ``num_retries=0``: the retry budget is ours.

        ``max_tokens`` gets headroom: a reasoning model spends the budget on
        its reasoning before the structured answer, and at the 2048 default the
        shipped model returned NO content for a real chunk's request — measured
        on conv-30, every chunk decoded to nothing. The sidecar this decoder
        replaced carried the same floor for the same reason.
        """
        settings = self._settings
        model = settings.extraction_model or settings.llm_model
        return build_lm(
            model,
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
            temperature=settings.llm_temperature,
            max_tokens=max(settings.llm_max_tokens, 4096),
            num_retries=0,
            **_openrouter_request_options(model),
        )

    @property
    def adapter(self) -> Any:
        """The pinned ``dspy.JSONAdapter`` every call of this decoder goes through."""
        return self._adapter

    @property
    def abstentions(self) -> Counter[str]:
        """What the response gates have dropped so far, by gate name (FR-017)."""
        return self._abstentions

    @property
    def nlp(self) -> Any:
        """The shared spaCy pipeline — the same resident model the local path loads."""
        return load_spacy_model(self._settings.spacy_model)

    def parse(self, text: str) -> Any:
        """The spaCy ``Doc`` for *text*, memoised for the segment's repeat callers.

        A one-entry memo, as on the local decoder: the frame-candidate and
        modality passes ask for the same chunk consecutively.
        """
        cached = self._doc_cache
        if cached is not None and cached[0] == text:
            return cached[1]
        doc = self.nlp(text)
        self._doc_cache = (text, doc)
        return doc

    def extract(
        self,
        text: str,
        extra_entity_labels: Sequence[str] | None = None,
        extra_relation_labels: dict[str, str] | None = None,
        speaker: str = "",
        *,
        frame_candidates: Sequence[FrameCandidate] = (),
    ) -> ExtractionResult:
        """Label one chunk. Exactly one provider call, or none for empty text.

        A section switched off in settings is left out of the request the same
        way an empty one is: not asked for, and never substituted locally.
        """
        if not text.strip():
            return ExtractionResult()
        settings = self._settings
        request = DecodeRequest(
            text=text,
            entity_labels=_entity_labels(extra_entity_labels) if settings.decode_entities else (),
            relation_spec=dict(extra_relation_labels or {}) if settings.decode_relations else {},
            frame_candidates=tuple(frame_candidates) if settings.decode_frames else (),
            speaker=speaker,
        )
        if not any((request.entity_labels, request.relation_spec, request.frame_candidates)):
            log.debug("every decode section is switched off; no request made for this chunk")
            return ExtractionResult()
        gated = ResponseGates(request, settings.decode_confidence_min).apply(
            self._call(request).decoded()
        )
        self._abstentions.update(gated.abstentions)
        return gated.result

    def extract_batch(
        self,
        texts: list[str],
        extra_entity_labels: Sequence[Sequence[str]] | None = None,
        extra_relation_labels: Sequence[dict[str, str]] | None = None,
        speakers: Sequence[str] = (),
        *,
        frame_candidates: Sequence[Sequence[FrameCandidate]] = (),
    ) -> list[ExtractionResult]:
        """One call per chunk, issued concurrently under the configured bound.

        Per-chunk failures are contained here, exactly as on the local path: a
        chunk that raises costs its own result, never the batch's.
        """
        if not texts:
            return []

        def _one(index: int) -> ExtractionResult:
            try:
                return self.extract(
                    texts[index],
                    extra_entity_labels[index] if extra_entity_labels else None,
                    extra_relation_labels[index] if extra_relation_labels else None,
                    speakers[index] if index < len(speakers) else "",
                    frame_candidates=frame_candidates[index]
                    if index < len(frame_candidates)
                    else (),
                )
            except Exception as exc:
                log.warning("decode failed on chunk %d (%s); skipping", index, exc)
                return ExtractionResult()

        with ThreadPoolExecutor(max_workers=self._settings.decode_concurrency) as pool:
            return list(pool.map(_one, range(len(texts))))

    def _call(self, request: DecodeRequest) -> DecodeResponse:
        """The one structured request this chunk gets, through the pinned adapter.

        ``format`` and ``parse`` are driven here rather than through the
        adapter's ``__call__``, which re-asks the provider in plain JSON mode
        when an answer does not parse. FR-010 allows no second call, so the
        rule is a property of this method rather than of a dspy branch.
        """
        signature = _signature_for(request)
        completions = self._lm(
            messages=self._adapter.format(signature, [], _inputs_for(request, signature)),
            response_format=_response_format(signature),
        )
        return DecodeResponse.model_validate(
            self._adapter.parse(signature, _completion_text(completions[0]))
        )
