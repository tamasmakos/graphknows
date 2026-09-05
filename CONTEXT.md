# GraphKnows

Agentic memory: a knowledge graph built from a conversation or document stream, and
hybrid retrieval over it. This glossary fixes the words the project uses for its own
concepts, so that a term means one thing in the code, the docs and the tracker.

## The semantic stack

**Taxonomy**:
The hierarchical skeleton of "is-a" relationships. Answers broad questions without
enumerating members.
_Avoid_: hierarchy, class tree

**Ontology**:
The formal blueprint: the classes, properties and logical rules that say what may be
related to what. Distinct from the taxonomy, which carries only subsumption.
_Avoid_: schema, vocabulary (when the formal blueprint is meant)

**Knowledge graph**:
The populated instantiation of the ontology — real entities and their real connections
as nodes and edges.
_Avoid_: the graph database, the store (those are the storage layer, not the model)

## The two layers

The project is built after the Tensor Brain model; where a module maps onto one of these
concepts it is named after it.

**Representation layer**:
The subsymbolic working state — embeddings and the fused candidate set. What is currently
active, not what is permanently known.
_Avoid_: the vector layer, working memory

**Index layer**:
The symbolic side — the concepts, predicates and time instances that can be named. A
symbol here is indexed by the embedding of its definition: the embedding *is* the
connection weight, not a side-car lookup.
_Avoid_: the symbol table, the registry

**Concept index / predicate index / episodic index**:
The three index types: entities, classes and attributes; relations; time instances. A
temporal fact earns its own episodic symbol rather than a timestamp column.

**Bottom-up decoding**:
Ingest: labelling raw input with symbols.
_Avoid_: encoding, tagging

**Top-down encoding**:
Retrieval: an activated symbol re-activating the working state and pulling back everything
attached to it. A channel that only decodes and is never encoded from is dead weight.

## Memory and its lifecycle

**Memory**:
The single transport-neutral entrypoint for every memory operation. The library API, the
MCP server and any adapter all go through it, so every transport sees identical behaviour.
_Avoid_: the client, the service, the engine

**Namespace**:
The hard multi-tenant isolation boundary. Each namespace is its own physical database, so
a query in one physically cannot see another's data.
_Avoid_: tenant, workspace, collection

**Scope**:
The identity triple — user, agent, run — that narrows an operation within a namespace. A
scope filters; a namespace isolates. Confusing the two is a data-leak bug.
_Avoid_: session key, filter, context

**Turn**:
One message in a conversation, as received.

**Chunk**:
The unit of stored, embedded, retrievable text. Several turns may share one.
_Avoid_: passage, document, segment

**State**:
Where a record sits in the consolidation lifecycle: `raw` when first buffered,
`consolidated` after a flush. Short-term and long-term memory are this lifecycle on one
graph, not two stores.
_Avoid_: STM/LTM as if they were separate databases

**Flush**:
Consolidating a session in place: draining the buffered turns through extraction, then
resolving duplicate entities and recomputing analytics.
_Avoid_: commit, promote, sync

**Hit**:
One ranked memory returned by recall, carrying its own provenance and raw timestamp.
_Avoid_: result, match, document

## Extraction

**Mode**:
Which decoder runs. `llm_free` runs the local decoder — deterministic, no API key.
`llm_assisted` runs the LLM decoder — one structured request per chunk over an injectable
ontology, and reaches the network. The two are alternatives behind one seam, never a layer
on top of each other: no mode falls back to the other decoder per chunk.
_Avoid_: profile, tier, preset

**Worth extracting**:
The per-turn linguistic judgement of whether a turn should mint entities at all. A skipped
turn is still stored, embedded and retrievable — a wrong skip costs graph coverage, never
evidence.
_Avoid_: filtering, relevance

**Abstention**:
A deliberate decline to emit — the extractor had nothing it could stand behind. Every
abstention is counted and surfaced; an abstention nobody can count is indistinguishable
from a broken extractor. A decoder call that fails after its retry budget is an abstention
too — counted under its own gate, distinct from the decoder returning nothing, and never a
new term.
_Avoid_: skip, drop, failure

**Grounding**:
An edge binding an extracted mention to a symbol in the index layer, carrying its full
provenance: channel, version, matcher, score, rank, chunk and span.
_Avoid_: link, mapping, annotation

**The anchor law**:
A candidate must come from the lexical index, so a purely-cosine match can never become a
grounding. Cosine similarity may produce retrieval tags and nothing more.

**Canonical predicate**:
The stable identity a relation is deduplicated on, chosen by a fixed ladder of decreasing
confidence and ending in abstention. Distinct from the *specific predicate*, which keeps
the surface wording.
_Avoid_: normalized relation, predicate type

**Polarity and modality**:
Whether a fact is asserted or negated, and with what epistemic force. "She said she might
move" and "she moved" are different facts, and the graph must not collapse them.
_Avoid_: negation flag, confidence

## Symbolic channels

**Channel**:
One lexico-semantic resource behind a common interface — FrameNet, WordNet, an imported
ontology. Adding a resource should be one package plus one registry line.
_Avoid_: source, provider, plugin, backend

**Frame**:
A schematic situation type with named participant roles.

**Frame element**:
One named role in a frame. Its coreness constraints gate whether a filled role
configuration is structurally valid.
_Avoid_: slot, argument, field

**Lexical unit**:
A word sense that evokes a frame. The **trigger** is the token in the text that fired it.

**Synset / sense**:
A WordNet concept, and one word's specific meaning within it. Relations split across the
two levels and the split is enforced — a sense-level claim written at synset level is
factually wrong.

**Digest**:
The compact projection of an ontology that the extractor is actually offered, as opposed
to the full source graph.
_Avoid_: summary, subset

## Retrieval

**Hybrid retrieval**:
The fused result of the retrieval channels — vector ANN, graph traversal and BM25 — rather
than any one of them.
_Avoid_: search, RAG

**Fusion**:
Combining ranked lists by reciprocal rank. Fusion is the attention side of the model,
averaging over candidates; resolving to a single entity is the sampling side. Both are
kept.
_Avoid_: blending, merging, ensembling

**Evidence recall**:
The retrieval metric decisions are made on. Headline accuracy is confirmation only; it is
too noisy to decide anything.

**The audit**:
The stratified manual inspection of relation edges across four binary axes — span,
predicate, direction, polarity-and-modality. It is the primary graph-quality metric,
because this work is generation-bound and retrieval metrics provably cannot see it.
_Avoid_: eval, benchmark (those mean the retrieval harness)

## Delivery

**Air-gapped**:
The runtime promise: after the image is built, the default path makes no network call
except to the graph database. Enforced by configuration and proven with egress blocked —
not a convention held per call site.
_Avoid_: offline, local-only, private, data-sovereign (a weaker and different claim)

**The contract surface**:
What the version guarantee actually covers: the `Memory` entrypoint, the MCP server,
extraction in both modes and injectable ontology. Everything else ships but is unsupported.
_Avoid_: the public API, supported features

**Re-ingest**:
The upgrade path. A schema or content-hash change re-mints node identity, so an upgraded
library refuses a stale graph rather than migrating it, and the source of truth is kept
outside the graph.
_Avoid_: migration, upgrade-in-place
