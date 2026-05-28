---
title: Investigative Agent Workflow — Evidence-Led Prototype — 2026-05-10
created: 2026-05-10
updated: 2026-05-10
type: concept
tags: [architecture, agent, langgraph, planning, graph, pipeline]
sources: [investigative-agent-architecture-redesign-2026-05-10, langgraph-agent-architecture-2026-05-08]
confidence: high
---

# Investigative Agent Workflow — Evidence-Led Prototype — 2026-05-10

## Goal

Refine the 3 macro-phase redesign into a more investigative, evidence-led workflow suitable for policy analysis, political research, and investigative journalism.

This note operationalizes seven required upgrades:
1. evidence ledger in state
2. dedicated claim-verification nodes
3. mandatory hybrid retrieval
4. contradictions as first-class outputs
5. parallel evidence threads
6. provenance metadata everywhere
7. session graph as a bounded hypothesis graph

## Diagnosis of current architecture

The current LangGraph agent in `services/agent/langgraph_agent.py` remains a tool-centric loop:
- `search_web` and `search_eu` collect documents into `_session_docs`
- `graphgen` enriches the session graph after search
- the final answer is synthesized directly from the same conversational loop

This is already stronger than a bare ReAct agent, but it still has structural limitations:
- no explicit evidence ledger; documents are accumulated but not scored or indexed by investigative thread
- no dedicated claim object or claim-verification pass
- no hard requirement that important claims combine graph paths with supporting passages
- contradictions are surfaced informally, if at all, rather than stored as durable outputs
- deep investigations still depend on a mostly serial tool loop
- provenance is partially present in docs/chunks/graph payloads but not normalized into a shared contract
- the Memgraph session graph is useful, but its role as a non-authoritative hypothesis graph is not explicit in state contracts

## Design principle

The workflow should be **evidence-led, not tool-led**.

That means the central objects are not just messages and tool outputs, but:
- sub-questions
- evidence threads
- ledger entries
- verified claims
- contradiction records
- provenance envelopes
- a bounded hypothesis graph

## Proposed macro architecture

Keep the 3 macro-phases, but strengthen the internal structure:

```text
START
  -> Scope
     - scope_frame
     - scope_map
     - scope_plan
  -> Investigate
     - investigate_dispatch_threads
     - investigate_collect_thread_evidence
     - investigate_trigger_graph_build
     - investigate_extract_claims
     - investigate_verify_claims
     - investigate_surface_contradictions
  -> Communicate
     - communicate_report
     - communicate_writeback
END
```

## 1. Evidence ledger in state

### Why
`collected_docs` is too weak for investigative work. The system needs to remember not only that a document was seen, but why it matters and how strong it is.

### Proposed contract
Add `evidence_ledger: list[EvidenceRecord]` to state.

Each `EvidenceRecord` should track:
- `evidence_id`
- `thread_id`
- `sub_question_id`
- short summary
- source authority score
- independence score
- directness score
- recency score
- linked claims
- tags
- provenance envelope

### Scoring semantics
These fields should remain simple and explainable:
- **source authority**: institutional weight / domain trust
- **independence**: whether the source is primary vs derivative / duplicated narrative
- **directness**: how directly the evidence addresses the sub-question
- **recency**: temporal freshness relative to the investigation

### Why this matters
This enables:
- better stopping criteria for gather loops
- more disciplined report synthesis
- explicit ranking of evidence quality
- explainable verification later in the flow

## 2. Dedicated claim verification node family

### Why
Verification should not happen implicitly inside final answer generation.

### Proposed nodes
- `investigate_extract_claims`
- `investigate_verify_claims`
- `investigate_surface_contradictions`

### Claim contract
Each `ClaimRecord` should track:
- `claim_id`
- claim text
- `sub_question_id`
- `status`: supported | contradicted | uncertain | unverified
- importance
- supporting evidence ids
- contradicting evidence ids
- graph path ids
- `requires_document_support`
- rationale

### Verification rule
For important claims, graph structure alone is insufficient.
A publishable claim must satisfy:
- status = `supported`
- at least one supporting evidence item with an exact passage + source document
- optional graph path / relation support to strengthen triangulation

## 3. Mandatory hybrid retrieval

### Why
Policy and investigative research needs both:
- **document grounding** for factual support
- **graph traversal** for relationships, paths, and gaps

### Rule
Important claims must never be justified by graph topology alone.

### Operational pattern
- use search / retrieval tools to find candidate documents and passages
- use graph path retrieval to expose relationship structure
- bind both to the same claim through provenance

### Implementation implication
Validation and reporting nodes should consume:
- evidence passage refs
- graph path refs
- extracted relation refs

and should reject graph-only support for high-importance claims.

## 4. Contradictions as first-class outputs

### Why
In this domain, unresolved conflict is itself a result.

### Proposed contract
Add `contradictions: list[ContradictionRecord]`.

Each contradiction should track:
- `contradiction_id`
- `claim_id`
- summary
- evidence_for_ids
- evidence_against_ids
- resolution_status: open | explained | needs_more_evidence

### Reporting rule
Contradictions should survive into final outputs under an explicit section, not be silently collapsed into a single “best answer”.

## 5. Parallel evidence collection by thread

### Why
Deep investigations usually branch into multiple evidence lanes:
- policy text
- actor mapping
- chronology
- funding / procurement
- criticism / opposition
- historical precedent
- cross-jurisdiction comparison

### Proposed contract
Add `evidence_threads: list[EvidenceThread]`.

Each thread tracks:
- `thread_id`
- title
- `sub_question_id`
- source classes
- status
- evidence ids
- notes

### Workflow implication
The investigate phase should dispatch several structured lanes in parallel, then merge them into the common evidence ledger.

This is the main mechanism for increasing depth without turning the whole workflow into one giant serial loop.

## 6. Provenance metadata everywhere

### Why
Auditable outputs require durable links from conclusions back to raw evidence.

### Proposed provenance envelope
Every evidence item should carry:
- retrieval event ref
- document ref
- optional passage ref
- optional relation ref
- optional graph path ref

This means the system can trace:
- relation -> chunk -> document -> retrieval event
- claim -> evidence -> provenance bundle
- report paragraph -> claim -> evidence

### Design choice
Keep provenance as a metadata envelope rather than polluting the core session graph with all audit detail.

## 7. Session graph as a bounded hypothesis graph

### Why
The Memgraph session graph should help the investigation reason, but it should not be confused with canonical truth.

### Proposed contract
Add `hypothesis_graph` to state with:
- `status`
- `graph_task_id`
- `graph_namespace`
- `seeded_from_corpus`
- `non_authoritative`
- `last_updated_at`

### Rules
- it is session-scoped
- it may be seeded from the corpus graph
- it is revisable as evidence changes
- it is explicitly non-authoritative
- only validated outputs are candidates for write-back or corpus promotion

## Prototype artifacts added

Code prototype added under `services/agent/graph/`:
- `state.py`
- `workflow.py`
- `__init__.py`

Focused test coverage added:
- `tests/test_investigative_workflow_prototype.py`

The prototype demonstrates:
- typed state contracts for evidence ledger, claims, contradictions, and hypothesis graph
- node map for the redesigned workflow
- grounding rule that rejects graph-only support for important claims
- required stream events for frontend observability (`phase_change`, `thread_started`, `evidence_added`, `graph_building`, `graph_ready`, `claim_verified`, `contradiction_detected`, `report_ready`)

## Recommended implementation slices

### Slice 1 — state contracts + non-invasive instrumentation
Add the typed `InvestigativeState` alongside the current runtime without switching execution yet.

Acceptance criteria:
- state module is importable in production code
- query/session routes can emit phase/event metadata without changing answer quality
- no regressions in current LangGraph path

### Slice 2 — evidence ledger population
Refactor `search_eu`, `search_web`, and graphrag retrieval outputs into normalized ledger entries.

Acceptance criteria:
- every retrieved evidence item includes scoring fields and provenance envelope
- evidence items are linked to sub-questions / threads
- raw document pile no longer acts as the only investigative memory

### Slice 3 — claim extraction + verification nodes
Introduce dedicated verification nodes and claim objects.

Acceptance criteria:
- claims are extracted before final report synthesis
- high-importance claims require passage-backed support
- unsupported claims are marked uncertain rather than written as facts

### Slice 4 — contradiction pipeline
Persist contradictions explicitly and expose them to UI/reporting.

Acceptance criteria:
- contradictions are represented as durable state
- final output includes unresolved conflicts section
- validation loop can send the workflow back to gather when evidence is thin

### Slice 5 — parallel thread dispatcher
Replace the single gather loop with structured parallel evidence threads.

Acceptance criteria:
- thread-level progress visible in stream events
- evidence ledger entries carry thread ids
- multiple source-class lanes can run concurrently without losing provenance

### Slice 6 — hypothesis graph hardening
Formalize the session graph as non-authoritative and bounded.

Acceptance criteria:
- state explicitly marks graph non-authoritative
- graph writes preserve provenance refs where possible
- report generation uses graph only in combination with source support for important claims

## Suggested Kanban follow-up graph

- `design: evidence-led investigative workflow redesign`
- `feat: investigative state contracts and stream events`
- `feat: evidence ledger normalization from search + graphrag`
- `feat: claim extraction and verification nodes`
- `feat: contradiction tracking and report section`
- `feat: parallel evidence thread dispatcher`
- `feat: bounded hypothesis-graph semantics and provenance integration`

## Links

[[investigative-agent-architecture-redesign-2026-05-10]] [[langgraph-agent-architecture-2026-05-08]] [[service-design-agent]] [[kanban-plan]]
