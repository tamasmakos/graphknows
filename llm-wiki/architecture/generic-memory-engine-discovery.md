---
title: Generic Memory Engine Discovery
created: 2026-05-22
updated: 2026-05-22
type: architecture
tags: [architecture, memory, agent, retrieval, evaluation]
confidence: high
---

# Generic Memory Engine Discovery

## Question

Can Graphknows act as a **generic agent memory engine** for Hermes-like sessions if it stores:

- raw trace as evidence
- distilled `State` / `Episode` / `Fact` objects as memory
- cross-session promoted facts as durable memory

rather than treating raw transcript and terminal output as the primary retrieval payload?

## Why this note exists

Earlier Hermes-memory planning drifted too far into Hermes-specific hook design. The real discovery question is more general:

> what is the smallest memory model that makes Graphknows useful across agents, while keeping terminal/tool exhaust out of the main memory surface?

This note records the stronger v2 prototype that answered that question with real session data.

## Prototype evidence

Research artifacts:
- `research/proto_generic_memory_v2.py`
- `research/proto_generic_memory_v2_results.json`

Real Hermes source sessions used from `/opt/data/state.db`:
- `20260522_182600_af16e04b` — initial memory plan
- `20260522_184057_8e8381` — prototype + plugin thinking
- `20260522_143801_b07242` — large memory-layer architecture/debugging session

Observed source sizes:
- 24 messages
- 127 messages
- 208 messages

## What the v2 prototype actually did

The prototype built three ingestion/retrieval lanes over the same real session set:

1. **raw lane**
   - compact near-raw assistant step trace + attached tool output snippets
2. **step lane**
   - deterministic step summaries
3. **distilled lane**
   - session `State`
   - session `Episode`
   - session `Fact`
   - cross-session promoted facts
   - cross-session state

It then ingested those lanes into Graphknows and compared recall quality.

## Key discovery 1 — `assistant.tool_calls` is mandatory

Hermes `messages.tool_name` is null for these sessions. Recovering tool identity from tool-result payload alone is incomplete.

The useful signal comes from **assistant rows with `tool_calls` populated**:
- tool name
- arguments
- call order

Therefore any real Hermes adapter or offline parser must read **both**:
- assistant `tool_calls`
- following tool result messages

A tool-result-only parser loses crucial structure.

## Key discovery 2 — the right unit is the assistant step, not the user turn

A better segmentation rule than user-turn boundaries is:

- one or more adjacent assistant messages
- plus the following batch of tool messages
- until the next user message or the next assistant block without attached tool work

This step unit matched the actual work rhythm of the sessions much better than user-turn segmentation.

Observed step counts from the three source sessions:
- 11
- 54
- 103

This is the right granularity for memory distillation.

## Key discovery 3 — distilled retrieval is materially better than raw retrieval

The v2 prototype computed a lightweight `useful_hit_rate` across four real query intents:
- continuity
- terminal-handling policy
- recurring-problem pattern
- adapter-contract query

Results:
- raw lane: `0.3333`
- step lane: `0.6667`
- distilled lane: `0.8333`

**Distilled vs raw delta: +0.5**

This is the strongest quantitative result from the prototype.

The raw lane mostly surfaced:
- transcript fragments
- tool dumps
- planning chatter
- path-heavy traces

The distilled lane surfaced:
- current working focus
- terminal filtering rule
- flush/relation failure pattern
- adapter lifecycle constraints

That validates the core architectural direction:

> raw trace is evidence; distilled objects are memory.

## Key discovery 4 — cross-session promotion works

The prototype automatically promoted facts that were supported by at least two sessions.

Promoted facts recovered automatically:
- **Terminal filtering requirement**
- **Flush relation bug**
- **Step segmentation rule**

Promoted-fact coverage metric:
- `1.0`

That means every promoted fact in the prototype had genuine multi-session support.

This is the first strong evidence that Graphknows can promote durable memory from repeated session evidence without hand-writing the final outputs.

## Key discovery 5 — distilled memory objects do not need GLiNER2

The first prototypes reused GLiNER2 on memory summaries. That pulled the memory-engine question back into extraction-model noise.

The stronger v2 prototype deliberately used a **deterministic lightweight extractor** for already-distilled memory objects.

Why this matters:
- distilled objects are already concise and typed
- they do not need full open-text NER behavior
- GLiNER2 is valuable for document/prose extraction, but it is not the right default for every memory object
- using a lightweight extractor for memory objects reduced latency and removed many shell-token artifacts

Architectural implication:

> Graphknows should support different extraction policies for different memory layers.

Suggested rule:
- raw documents / unstructured content -> GLiNER2 + relation extraction
- distilled memory objects -> deterministic lightweight extraction or typed metadata only

## Key discovery 6 — flush worked after proper namespace hygiene

The v2 prototype also validated STM→LTM promotion on the distilled lane.

After clearing existing LTM rows for the prototype namespace first, flush produced:
- `nodes_written = 86`
- `edges_written = 207`
- `chunks_promoted = 28`
- `turns_promoted = 0`
- `errors = []`

This matters because earlier results were misleading when the namespace already existed: `ON CONFLICT DO NOTHING` hid the real promotion counts.

Operational discovery:
- for prototype/eval work, always clear the LTM namespace before interpreting promote counts
- pgvector pool ownership must stay aligned with the loop context used by the flush path

## Key discovery 7 — the generic data model is now justified

The prototype supports a minimal generic model:

### RawEvent
Adapter-facing, append-only evidence.

Fields needed in practice:
- `session_id`
- `message_id`
- `role`
- `event_type`
- `content`
- `timestamp`
- `parsed_metadata`

### StepSummary
The first distillation boundary.

Fields needed in practice:
- assistant ids
- tool ids
- tool names/types
- summary
- evidence refs

### MemoryItem
The retrieval-facing unit.

Kinds validated in the prototype:
- `state`
- `episode`
- `fact`
- `promoted_fact`
- `cross_session_state`

This is thinner and more reusable than a Hermes-native turn model.

## Recommended architectural direction

### 1. Keep a trace layer
Append-only raw events for:
- provenance
- replay
- debugging
- re-distillation

### 2. Add a distillation layer
Graphknows-native logic should produce:
- `State`
- `Episode`
- `Fact`
- promoted cross-session facts

### 3. Retrieve from distilled memory by default
Default recall should prefer:
- continuity -> state + recent episodes
- fact lookup -> facts + promoted facts
- pattern lookup -> promoted facts + episodes

Raw trace should be a secondary/audit lane, not the primary memory lane.

### 4. Treat extraction policy as layer-specific
- document memory != distilled memory
- Graphknows should not force the same entity-extraction path for both

## Minimum adapter contract implied by the discovery work

This is still provisional, but the prototype makes the direction clear.

A thin adapter should be able to do roughly five things:
- append raw events
- delimit step boundaries
- write/update current state
- append episodes/facts/promoted facts
- query memory by lane and intent

Put differently, the adapter should supply events and consume recall. The distillation policy should live in Graphknows, not in the framework adapter.

## Remaining open questions

These are the real discovery questions still left before proper planning:

1. **When to distill**
   - per tool completion?
   - per assistant step?
   - per session switch/end?
   - likely: step summaries online, promotion offline

2. **How to promote facts**
   - repeated occurrence alone is not enough
   - need support + stability + usefulness + scope

3. **How to model open loops**
   - open issue / blocker / next action deserves first-class treatment
   - may need a dedicated `OpenLoop` kind or richer `State`

4. **How much raw trace to retain**
   - enough for evidence and replay
   - not enough to pollute recall by default

5. **How to calibrate retrieval**
   - the distilled lane is clearly better, but current scoring is still simple
   - likely next step: typed retrieval lanes before expensive reranking

## Decision-quality conclusion

The deeper prototype is good enough to support a design decision:

> Graphknows should evolve toward a generic memory-engine architecture where raw events are evidence and distilled memory objects are the primary retrieval surface.

That direction is now grounded in:
- real Hermes session data
- multi-session automatic distillation
- automatic cross-session promotion
- real Graphknows ingest
- real STM/LTM queries
- a quantified raw-vs-distilled retrieval comparison

## Related pages
- [[architecture/memory-layer-v2]]
- [[architecture/product-architecture]]
- [[ops/technical-debt]]
