# Generalisation audit — from LoCoMo memory to universal neurosymbolic memory

Date: 2026-09-08. Baseline commit: `fb3b104`. Method: eight parallel read-only audits
(ingestion, storage, symbolic, entity resolution, extraction, retrieval/eval, Claude
integration, Claude Code hook facts), consolidated here. Every finding carries a
`file:line` anchor from the audited tree.

## 1. Verdict

The repo has Tensor-Brain vocabulary and a LoCoMo body. The design doc says "a layer that
writes but is never read is dead weight"; four planes are write-only today (WordNet senses,
topics, pagerank/community/graph-embedding columns, the FE/SEMTYPE graph mirror). Of the
three symbolic channels, only plain FrameNet runs by default; the ontology channel, the one
with a real taxonomy walk, is off. The turn-line regex `Speaker: text` is assumed in
parsing, extraction, hygiene, deixis and BM25, so a YAML key, a type annotation or a
citation is misread as a speaker on non-dialogue text.

Root cause is not any single module. Three seams lack a neutral data model, and every
LoCoMo assumption leaked through them:

| Seam | What exists | What is missing |
|---|---|---|
| Input | `str` or `[{role, content}]` (`memory.py:389`) | `Source → Segment` with kind, byte range, MIME, structure path |
| Extraction record | per-extractor dicts, spans dropped after `_dedupe_entities` | `Mention / Fact / Event` with offsets, extractor id+version, polarity, validity |
| Domain knowledge | settings flags + `if enable_x` ladder (`channels/registry.py:28-56`) | `DomainPack` bundling ontology, lexicon, frames, labels, hygiene, channel, expander |

Fix the seams and the rest becomes swapping data, not editing code.

Evidence note: `evaluation/results/` holds LoCoMo only (best committed run: accuracy 0.869,
evidence recall at probe 0.95, n=191). No LongMemEval or BEAM results are committed. The
only BEAM number in the tree is a code comment (`retriever.py:127-131`, 0.371→0.426 with
`neighbor_radius=1`, which costs LoCoMo 0.895→0.814). Both harnesses also feed whole
sessions as one text blob, never turn-fed (`evaluation/longmem/dataset.py:84-101`,
`evaluation/beam/dataset.py:128`), so speaker, STM and temporal machinery never engaged on
them. The "did terribly" result is partly a harness artefact and is currently unreproducible.

## 2. Issue register

Severity: B = blocker for universality, M = major, m = minor.

### 2.1 Ingestion and parsing

| # | S | Issue | Evidence |
|---|---|---|---|
| I1 | B | Only two input shapes; only one parser (`PlainTextParser`). No code, PDF, JSONL transcript, table parser | `ingestion/parsers/` has `text.py` + `_sentences.py`; `ingest.py:496` hardcodes the parser |
| I2 | B | `Role` is a closed 4-value enum; tool name, args, file paths, nesting have no field | `models/message.py:14, 64` |
| I3 | B | Chunking is Markdown-heading or sentence only; fenced code blocks not special-cased; a `.py` file is one blob or spaCy-mangled | `parsers/text.py:247, 298`; `HEADING_RE` |
| I4 | M | Provenance thin: `char_offset` found by first-line search and silently 0 on miss; no MIME, git commit, full-content hash; `corpus_ingest` ignores `id` for dedup | `text.py:150-181, 287`; `memory.py:1068` |
| I5 | M | Dialogue heuristics inside the core: 5-turn window stride 3, `speaker: body` search surface, deixis ≤2 speakers, CCO typing classes with no code/paper classes | `ingest.py:358-360, 1165, 1174-1237, 84, 116`; `text.py:38` |
| I6 | M | No `Parser` protocol or registry; `ingest_parsed` exists but is unreachable from the public API | `ingest.py:486-499`; `memory.py` never routes to it |
| I7 | M | Hard undocumented size ceiling; BEAM needed a bespoke splitter in the harness ("a single oversized ingest crashes the extraction model") | `evaluation/beam/dataset.py:65` |
| I8 | M | No content-hash dedup, no resumable cursor, `corpus_ingest` reconnects the store per document | `memory.py:440-464, 1068` |
| I9 | m | `_ingest_parsed` and `_write_extracted` are 6-responsibility orchestrators every new input type must thread through | `ingest.py:509-653, 1296-1416` |
| I10 | m | Undated text gets no temporal anchor, silently | `ingest.py:1078, 2504-2513` |

### 2.2 Storage schema

| # | S | Issue | Evidence |
|---|---|---|---|
| S1 | B | No migration; any DDL edit strands every database | `_schema.py:438-452` docstring "There is no migration" |
| S2 | B | TOPIC / IN_TOPIC written, never read by retrieval | `topics/persist.py`; only `cli/topics.py`, `mcp/tools/topics.py` read it |
| S3 | B | ENTITY.pagerank / community_id / graph_embedding: three whole-graph passes per flush, zero ranking readers | `graph_store.py:3523-3706`; `channels/ontology.py:26` notes the signal regressed retrieval |
| S4 | M | FE / SEMTYPE / HAS_FE / REQUIRES_FE / EXCLUDES_FE graph plane is a write-only mirror of a Python lookup | `_schema.py:150-153` |
| S5 | M | Two representations of "chunk evokes frame": FRAME→EVOKED_BY→CHUNK and FRAME_INSTANCE→FRAME_EVOKED_IN/EVOKES | `graph_store.py:1424, 2425` |
| S6 | M | `REL.evidence`, the one load-bearing provenance property, is undeclared in DDL | `graph_store.py:940, 1253, 3362` vs `_schema.py` |
| S7 | M | Conversation spine (SESSION/TURN, CHUNK.speaker, CHUNK.kind) carried into every document/code chunk as empty columns; FILE is a second parallel "source" | `_schema.py:31-94` |
| S8 | M | No extractor version, ingestion timestamp, or namespace marker on vertices; namespace is only physical DB-per-tenant | `_schema.py` |
| S9 | m | `REL.reified`, `REL.anchor` have no live reader | `graph_store.py` write paths only |
| S10 | m | `GraphStore` is a 4066-line god object with ten responsibilities | section dividers at `graph_store.py:569…4018` |

### 2.3 Symbolic layers

| # | S | Issue | Evidence |
|---|---|---|---|
| Y1 | B | WordNet is write-only: senses, synsets, hypernym chains minted every ingest; the reader (`WordNetChannel`) was deleted in `9cec590` | `ingest.py:1934-1972`; `graph_store.py:1858, 1929`; stale `channels/__pycache__/wordnet*.pyc` |
| Y2 | M | Ontology channel and frame facts off by default; top-down activation mostly does not run | `settings.py:426-438` |
| Y3 | M | Channel registry is a closed `if enable_x` ladder; no plugin discovery. `Memory(channels=[...])` works but is undocumented | `channels/registry.py:28-56` |
| Y4 | M | Eager full materialisation of ontologies; one-hop BROADER walk at query time; no transitive closure, no OWL restrictions, no rule inference | `loader.py:350-381`; `graph_store.py:3029-3041`; `rdf/taxonomy.py:49-60` |
| Y5 | M | Frame is FrameNet's data model, not a generic abstraction; gate constants tuned for "conversational filler"; `nltk` called directly | `framenet/fe.py:11-33`; `frame_srl.py:27-51` |
| Y6 | m | Three independent taxonomy walkers (`skos.py`, `rdf/taxonomy.py`, `graph_store._narrower_scores`); `skos.py` lexicalisation duplicates `lexical.py`; live path (`catalog.py`) uses neither | `skos.py:117, 134, 318, 444, 494` |
| Y7 | m | Predicates are free strings; no predicate index node aligned to ontology properties | `relations/filter.py:213-266` |
| Y8 | m | Ontology predicate annotation (`ontology_property`) is written, never read | `ingest.py:2006-2103` |

### 2.4 Extraction

| # | S | Issue | Evidence |
|---|---|---|---|
| X1 | B | Turn-line regex assumed in speaker detection, deixis, hygiene; `Name: text` in YAML/annotations/citations is treated as a speaker and can corrupt spans | `extractor.py:734-745, 365-444`; `_lingfeatures.py:46, 59-113`; `hygiene.py:104-109` |
| X2 | B | Hygiene gates reject code identifiers (`::`, `<>`, `{}`, `()`), long technical names (>8 words), lowercase coordinated terms, verb-headed titles | `hygiene.py:99, 103, 120, 235-254, 349-370` |
| X3 | B | 18-label conversational entity inventory always unioned in; ontology hints can add but never replace | `extractor.py:95-114`; `decoder.py:44-47, 208-214`; `ingest.py:1765` |
| X4 | M | `_SURFACE_ALIASES` social predicate table appears dead | `relations/filter.py:100-138` |
| X5 | M | Modality/hedge vocabularies are conversational; scientific hedges (propose, assume, conjecture) not covered | `linguistics.py`, `frame_srl.py` LVC lists |
| X6 | M | No injection point for a non-FrameNet frame source | `frame_srl.py:48` |
| X7 | M | English-only pinned models, no language routing | `settings.py:265-314` |
| X8 | M | `_relex` mixes schema build, deixis branching, two-pass invocation; `None` vs `{}` carry different meanings | `extractor.py:936-1020, 970-990` |
| X9 | m | Thresholds tuned on conv-26/conv-30, no per-domain override | `relex_threshold=0.35`, `relation_verifier_threshold=0.5` |
| X10 | m | Negation/modality computed only for FRAME_INSTANCE, never attached to REL; "did NOT observe X" stores as asserted | `linguistics.py` vs `write_relation` |

Keep as the seam: `LLMDecoder` + `DecodeRequest` + `ResponseGates` are already ontology-driven and domain-agnostic (`decoder.py`, `anchor.py`); `temporal.py` is library-driven.

### 2.5 Entity resolution

| # | S | Issue | Evidence |
|---|---|---|---|
| E1 | B | `resolve_entities` loads the whole ENTITY table into Python per flush; no blocking, no ANN | `entity_resolution.py:203-208`; `graph_store.py:3706-3730` |
| E2 | B | Embedding similarity explicitly unused (measured unsafe on nicknames); precision rests on plural-strip, head-noun, 3-char prefix + 2x dominance | `entity_resolution.py:25-30, 311-434` |
| E3 | B | No mention layer; merge is `DETACH DELETE` with no log, no split | `graph_store.py:834-865, 3971` |
| E4 | M | Type frozen at first sight (`coalesce(e.type, row.etype)`) | `graph_store.py:838` |
| E5 | M | Synsets and INSTANCE_OF never consulted as merge evidence or veto | `entity_resolution.py` imports neither `wordnet.py` nor `match.py` |
| E6 | M | Head-noun merge capped at 2000 parses per flush; tail never resolves | `entity_resolution.py:311-334` |
| E7 | m | Only blocking key in the system is a person-only 3-char prefix | `entity_resolution.py:374` |
| E8 | m | No ER metric (B-cubed, pairwise P/R); only end-to-end QA | `tests/ingestion/consolidation/` |
| E9 | m | Concurrent flushes race on the in-memory namespace snapshot | `entity_resolution.py:203` |

### 2.6 Retrieval and evaluation

| # | S | Issue | Evidence |
|---|---|---|---|
| R1 | B | LongMemEval and BEAM are never turn-fed; speaker/STM/temporal paths dead for them; no results committed | `evaluation/longmem/dataset.py:84-101`; `evaluation/beam/dataset.py:128`; `evaluation/results/` |
| R2 | B | No retrieval-time abstention signal; abstention is a generation-prompt sentence | `grep abstention graphknows/` hits only ingest; `longmem/prompts.py` stage 5 |
| R3 | M | Every knob pinned on conv-26/conv-30; `neighbor_radius` is proven to pull opposite ways per corpus | `retriever.py:127-138, 556-562`; `settings.py:344-361, 426-438`; `evaluation/common/config.py:39-45`; `frame_boost.py:92-95` |
| R4 | M | Top-down symbolic expansion only re-ranks an already-fused pool; cannot recover a candidate no channel found | `retriever.py:248-286` |
| R5 | M | No query planning: multi-hop, aggregation, before/after, "where is X defined", "what method" | `retriever.py:97-120, 176-246` |
| R6 | m | ~65 settings, 13 retrieval knobs with no named profile | `settings.py` |
| R7 | m | No per-category evidence recall, no channel ablation matrix, no non-QA eval | `evaluation/common/reporting.py`; `datamodels.py:67-73` unaggregated |

### 2.7 Claude and harness integration

| # | S | Issue | Evidence |
|---|---|---|---|
| C1 | B | No transcript parser for Claude Code JSONL (or any harness) | `ingestion/parsers/` |
| C2 | B | No hook scripts; only `graph-update.sh` for the code graph | `.claude/hooks/`, `.claude/settings.json` |
| C3 | M | Ingest cannot take a tool-result-shaped payload; tool name/args lost | `memory.py:389-391`; `models/message.py` |
| C4 | M | No idempotency by message uuid | `memory.py:413-430`; `bounds.py:47-54` |
| C5 | M | No per-fact forget; only session/namespace purge behind admin flag | `mcp/tools/admin.py:1-46` |
| C6 | m | `render_memories` is prompt-ready but wired only to LangGraph | `integrations/langgraph/_hooks.py:146-199` |
| C7 | m | torch/spacy/gliner in core deps; ~10 GB image; ~100 s cold start; not a light `.mcp.json` dependency | `pyproject.toml:28-64`; `Dockerfile`; `client/mcp.py:33-39` |
| C8 | m | No skill/CLAUDE.md teaching when to call memory; no HTTP transport for hosted clients | `docs/integrations.md`; `docs/services.md:114` |

## 3. Target architecture

### 3.1 Ingestion: Source → Segment

```python
@dataclass(frozen=True)
class Source:      # one file, session, URL, transcript
    id: str; uri: str; mime: str; content_hash: str; meta: dict

@dataclass(frozen=True)
class Segment:     # unit of embedding + extraction
    id: str; source_id: str; text: str
    kind: Literal["prose", "turn", "code", "table", "tool_call", "tool_result", "citation"]
    path: str                      # heading path, symbol path, page/section, turn index
    byte_range: tuple[int, int]
    role: str | None = None        # speaker / user / assistant / tool
    ts: str | None = None
    meta: dict = field(default_factory=dict)

class Parser(Protocol):
    mimes: frozenset[str]
    def parse(self, source: Source) -> Iterable[Segment]: ...
```

Parsers to ship: markdown/text (existing, plus fenced-code awareness), code via tree-sitter
(segment per function/class, `path` = qualified symbol, `calls`/`imports`/`defines` emitted
as facts), PDF/paper (page + section aware, citations as segments), Claude Code transcript
JSONL, JSON/CSV tables. Registry keyed by MIME, overridable via `Memory(parsers=[...])`.
Dedup by `content_hash` before any model runs. Resumable corpus jobs with a cursor.

### 3.2 Extraction: one record type, pluggable extractors

```python
@dataclass(frozen=True)
class Mention:  surface: str; type: str; span: tuple[int, int]; confidence: float; extractor: str
@dataclass(frozen=True)
class Fact:     subject: Mention; predicate: str; object: Mention; confidence: float
                polarity: bool; modality: str | None; valid_from: str | None; valid_to: str | None
                evidence_span: tuple[int, int]; extractor: str

class Extractor(Protocol):
    def extract(self, segment: Segment, pack: DomainPack) -> tuple[list[Mention], list[Fact]]: ...
```

`GLiNER2EntityExtractor` and `LLMDecoder` already satisfy this duck-typed; formalise it.
Add `CodeExtractor` (tree-sitter, no ML), `CitationExtractor`, `ToolCallExtractor`.
Hygiene becomes a `HygieneConfig` value with a `conversational` and a `generic` preset;
`dialogue_mode` inferred from the fraction of lines matching the turn regex.

### 3.3 Domain knowledge packs

```python
class DomainPack(Protocol):
    name: str
    def concepts(self) -> Iterable[OntologyTerm]: ...          # classes + properties, lazy
    def entity_labels(self) -> tuple[str, ...]: ...            # replaces, not unions
    def prompt_addendum(self) -> str: ...
    def hygiene(self) -> HygieneConfig: ...
    def frames(self) -> Iterable[Frame]: ...                   # generic Frame, see below
    def channel(self) -> Channel: ...                          # populate() + collect()
    def expand(self, symbol: str, direction: str, hops: int) -> Iterable[str]: ...
```

Generic frame: `Frame(name, definition, roles: [Role(name, coreness, semtype)])`. FrameNet
becomes one provider; a code pack provides `Call(caller, callee, args)`, `Define`, `Import`;
a paper pack provides `Claim(claim, method, evidence, citation)`. Packs are passed as
`Memory(packs=[...])`; the `if enable_x` registry ladder goes away. A pack with no working
`collect()` is visibly incomplete, which prevents the WordNet write-only failure by
construction. WordNet as a pack: `collect()` does hypernym-expanded lookup, `expand()` walks
synsets, and ER consumes `INSTANCE_OF`/`HAS_SENSE` as evidence. Large ontologies get an
iterator contract and a persisted index instead of eager dict materialisation.

### 3.4 Golden schema

```
SOURCE(id, uri, mime, content_hash, imported_at, namespace)
SEGMENT(id, text, kind, path, byte_start, byte_end, role?, ts?, embedding, extractor_version, created_at)
  SEGMENT -[:PART_OF]-> SOURCE ;  SEGMENT -[:NEXT]-> SEGMENT
ENTITY(id, name, name_norm, type_histogram, aliases, state, created_at)
CONCEPT(uri, label, definition, embedding, pack)             # class | synset | frame  (concept index)
PREDICATE(id, label, canonical, functional, pack)            # (predicate index)
EPISODE(id, start, end)                                      # (episodic index; intervals, not dates)
FACT(id, confidence, polarity, modality, valid_from, valid_to, is_current, extractor, evidence[])
  SEGMENT -[:MENTIONS {span, confidence, extractor}]-> ENTITY
  SEGMENT -[:EVOKES {confidence, roles}]-> CONCEPT
  ENTITY  -[:INSTANCE_OF {score, pack}]-> CONCEPT
  CONCEPT -[:BROADER {transitive}]-> CONCEPT
  ENTITY -[:SUBJECT_OF]-> FACT -[:OBJECT_IS]-> ENTITY ;  FACT -[:USES]-> PREDICATE
  FACT -[:ASSERTED_IN {span}]-> SEGMENT ;  FACT -[:VALID_DURING]-> EPISODE ;  FACT -[:CONTRADICTS]-> FACT
  ENTITY -[:MERGED_INTO {layer, evidence, at}]-> ENTITY       # merge log, reversible by replay
```

Dropped: SESSION/TURN as types (a session is a SOURCE of kind `session`; a turn is a SEGMENT
of kind `turn`), TOPIC/IN_TOPIC, pagerank/community_id/graph_embedding columns, the FE/SEMTYPE
mirror, the FRAME vs FRAME_INSTANCE duplication, free-string predicates. Add a versioned DDL
diff migration before anything else changes.

### 3.5 Entity resolution for millions of documents

1. Mention layer: keep spans on the MENTIONS edge; never collapse a surface into an alias string.
2. Blocking: `(name_key, type)` where name_key is normalised-prefix or phonetic; query the block, never the table.
3. ANN over entity definition embeddings (context sentence + type), gated by the existing floor+margin `symbolic/match.py:accept()`.
4. Symbolic evidence: same synset or ontology class supports a merge; disjoint sibling classes veto it.
5. Type histogram instead of first-wins; ontology adjudicates ties.
6. Deterministic canonical name: mention count, then length, then `created_at`.
7. `MERGED_INTO` log before any delete; split by replay.
8. Incremental: resolve only entities touched by this flush plus their block neighbours.
9. Measure first: B-cubed and pairwise P/R on a hand-labelled non-LoCoMo sample, false-merge rate, hygiene drop rate on code and papers, wall-clock at 100k / 1M / 10M entities.

### 3.6 Retrieval

- `RetrievalProfile` dataclass (neighbor_radius, window sizes, pool factors, date gate, frame alpha, channel toggles) selected by corpus shape: dialogue, long document, code/paper. The RRF spine stays fixed.
- Top-down expansion moves into candidate generation: pack `expand()` on query symbols feeds entity and concept collectors, not just the re-ranker.
- A retrieval-time abstention signal: empty or below-floor pool returns a typed "no evidence" result the generator can honour.
- A small query planner for multi-hop, aggregation and interval questions, reusing the same traversal (semantic recall sets the symbol, episodic recall samples it, per the Tensor Brain rule).

### 3.7 Claude integration

Hook events that can inject context: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PostToolUseFailure, Stop. PreCompact and SessionEnd can only observe.

| Hook | Payload used | Memory call | Output |
|---|---|---|---|
| SessionStart | `cwd`, `session_id`, `source` | recall "recent work in cwd" | `additionalContext` |
| UserPromptSubmit | `prompt` | recall top-k, render with existing `render_memories` | `additionalContext` |
| PostToolUse (Edit, Write, Bash) | `tool_name`, `tool_input`, `tool_response` | ingest a `tool_result` segment, summarised to the 64 KB cap | none |
| Stop | `transcript_path` | parse new JSONL records since last uuid checkpoint, ingest | none |
| PreCompact | `transcript_path` | full catch-up ingest, then flush | none |
| SessionEnd | `session_id` | flush STM → LTM | none |

Transcript parser: filter `type in {user, assistant}` and not `isMeta`; walk `message.content`
blocks (`text`, `thinking`, `tool_use`, `tool_result`); segment kind from block type; dedup
key = record `uuid`; timestamp, `cwd`, `gitBranch` become segment metadata. The docs warn the
format is internal and version-dependent, so the parser must be tolerant and versioned.

Hook process stays light: it talks to the running MCP server through the stdlib-only
`GraphKnowsMCPClient`, never imports torch. New modules:
`ingestion/parsers/claude_transcript.py`, `integrations/claude_code/hooks.py` (stdin JSON in,
`hookSpecificOutput` JSON out), a `memory_forget(fact_id | entity)` MCP tool, and a shipped
skill that tells the agent when to recall and remember. The same hook callbacks serve the
Claude Agent SDK Python API (PreToolUse, PostToolUse, UserPromptSubmit, Stop, PreCompact).

Settings block:

```json
{"hooks": {
  "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks recall"}]}],
  "PostToolUse":      [{"matcher": "Edit|Write|Bash", "hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks observe", "async": true}]}],
  "Stop":             [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks remember"}]}],
  "PreCompact":       [{"hooks": [{"type": "command", "command": "python -m graphknows.integrations.claude_code.hooks catchup"}]}]
}}
```

`.mcp.json`: `{"mcpServers": {"graphknows": {"command": "graphknows-mcp", "env": {"GRAPHKNOWS_MODE": "llm_free", "GRAPHKNOWS_NAMESPACE": "claude-code"}}}}`.

Packaging: move torch/spacy/gliner behind a `local` extra so the hook and client install stays small.

## 4. Roadmap

Measure before build, and delete before add.

1. **Evidence (1 week).** Turn-feed LongMemEval and BEAM, commit their results, add per-category evidence recall and a channel ablation matrix. Build a 200-cluster ER ground truth on a mixed corpus and record B-cubed. This is the baseline every later change is judged against.
2. **Delete dead weight (days).** Stop writing TOPIC, pagerank/community/graph_embedding, the FE mirror, WordNet senses (until a reader exists), `_SURFACE_ALIASES` if confirmed dead, the stale `.pyc`. Declare `REL.evidence`. Add a DDL migration mechanism.
3. **Seams (2 to 3 weeks).** `Source/Segment` + `Parser` protocol; `Mention/Fact` + `Extractor` protocol; `HygieneConfig` presets and inferred `dialogue_mode`; `RetrievalProfile`. Existing behaviour ported as the "conversational" profile so LoCoMo numbers hold.
4. **Packs (2 weeks).** `DomainPack` protocol; FrameNet, WordNet (with a real `collect()`), CCO/personal as packs; generic `Frame`; predicate index. Ontology channel on by default, A/B'd.
5. **Golden schema (2 weeks, behind a flag).** SOURCE/SEGMENT/FACT/PREDICATE/EPISODE; collapse FRAME duplication last, with evidence-recall regression gate.
6. **ER at scale (2 weeks).** Mention layer, blocking, ANN, symbolic evidence, type histogram, merge log, incremental resolve. Judged on step 1's ER metric.
7. **New inputs (1 week each).** tree-sitter code parser and extractor; Claude transcript parser and hooks; PDF/paper parser with citation frames. Each ships with a small non-QA eval (code symbol lookup, section retrieval).
