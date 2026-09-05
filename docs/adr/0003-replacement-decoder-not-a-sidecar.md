# 3. `llm_assisted` is a replacement decoder, not a sidecar

Status: accepted.

## Context

`llm_assisted` was a set of add-ons bolted onto the llm-free stack.
`build_entity_extractor()` always returned `GLiNER2EntityExtractor`, and
`build_relation_extractor()` wrapped it in a DSPy *sidecar* that contributed extra
relations when `GRAPHKNOWS_DSPY_RELATIONS` was on. The mode named a bundle of
capabilities, not a decoder.

That shape cost more than it bought. Two producers write relations into the same channel,
so a measured difference between the modes cannot be attributed to either; the local models
load in both modes, so the "assisted" arm pays GLiNER2, the DeBERTa verifier and the
transformer frame parser on top of the API call (measured on conv-30: the frame parser
alone was 12 s of a 32 s chunk); and DSPy sat behind an `[assisted]` extra that also carried
`rdflib`/`networkx`, so a mode of the documented contract surface raised an import-time
error on a plain install.

## Decision

`Memory(mode="llm_assisted")` **selects one of two decoders behind the same duck-typed
seam**. The LLM decoder issues one structured request per chunk over the closed
vocabularies the local decoder is handed today — entity labels, the chunk's ontology
property spec, the LU-trigger frame candidates with their core frame elements, the chunk's
speaker — and returns entities, relations with confidence, and frame-role instances as
verbatim surfaces that are re-anchored locally. GLiNER2, the DeBERTa verifier and the frame
parser are never constructed in that mode; spaCy runs in both, because spaCy is the parse,
not the decoder. The sidecar and its knob are deleted rather than left dormant.

A failed call **abstains** — its own gate name, distinct from a successful empty decode —
marks the chunk in the graph, and is re-driven by `Memory.redecode(session_id)`.

**Alternatives rejected:**

- *Layering (keep the sidecar, add LLM output on top of GLiNER2's).* Rejected: two
  producers on one channel make the release gate unreadable, and the local stack still
  loads, so the mode's cost is the sum of both rather than the price of one call.
- *Hybrid (LLM decoder with per-chunk fallback to the local one on failure).* Rejected: a
  run's graph would then be a silent mixture of two decoders in unknown proportion, which
  is the same unreadable comparison plus a hidden dependency on the local models that
  SC-003 says must not load. Failures are counted and re-driven instead.
- *Serial multi-call (entities → relations → verify).* Rejected in favour of one call: it
  multiplies latency and cost per chunk and reintroduces a verifier stage the confidence
  value already occupies the place of. It returns only as a fresh effort if the one-call
  shape fails the gate — the approach goes back to the map rather than being patched.

## Consequences

Good: an assisted run loads zero local extraction models, so the two arms differ by exactly
one thing and the gate (evidence recall ≥ `llm_free`, audit precision ≥ `llm_free` on all
four axes) means something; the three response sections switch off independently, so a
regression is attributable to a section rather than to "the LLM"; a decoder failure is
visible in the ingest report and recoverable, instead of silently thinning the graph.

Bad: **DSPy becomes a core dependency** — every install carries it, including air-gapped
ones that will never call a provider — and the **`[assisted]` extra is removed**, so
`pip install 'graphknows[assisted]'` breaks; users who installed it for the RDF/OWL stack
move to **`[ontology]`**, which is unchanged. `llm_assisted` **joins the contract surface**,
so it is held to the release gate from here on and can no longer be changed as an
experiment. **Per-chunk fallback is forbidden in every code path**, which means a provider
outage costs coverage on those chunks until they are re-driven; that is the price of a
comparison that can be trusted.

Neutral: the mode still opts out of the air-gap guarantee, as LLM-assisted extraction
already did. Everything downstream of extraction — resolution, groundings, frames,
temporal facts, consolidation, retrieval — is untouched, because both decoders return the
same `ExtractionResult`.
