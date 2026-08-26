# language: en
#
# memory_pipeline.feature — GraphKnows agentic memory service
# =============================================================================
# DESCRIPTIVE spec: what the pipeline does TODAY, verified against the code at
# HEAD. Sections:
#
#   A. Storage & schema     — one ArcadeDB database per namespace, one lifecycle
#   B. Vocabularies         — two ontologies, one FrameNet index
#   C. Ingestion            — chunking, two-pass extraction, graph writes
#   D. Consolidation        — the flush: drain, resolve, analytics, state flip
#   E. Retrieval            — four collectors, optional channels, RRF, one sort
#   F. NOT IMPLEMENTED      — intent on record; NONE of it is current behaviour
#
# STRUCTURE NOTE
#   Gherkin permits exactly one `Feature:` per file. Everything is therefore one
#   Feature subdivided with `Rule:` (Gherkin 6+), one Rule per business rule.
#
# STATUS NOTE
#   Nothing executes this file: the repo has no behave/pytest-bdd runner and no
#   step definitions. It is a reference document, and it is only worth having if
#   every scenario outside Section F is true of the code. Where a claim is
#   backed by a real test, the test path is named in a `# test:` comment.
#
# TAG LEGEND
#   @schema @ingest @extract @flush @retrieve   pipeline stage
#   @ontology @frames @temporal                 the layer under test
#   @arcadedb                                   asserts storage behaviour
#   @sql @cypher                                the query language actually used
#   @model                                      needs relex / spaCy / an embedder
#   @slow                                       minutes, not seconds
#   @doc                                        non-executable; recorded for review
#   @measured                                   the number quoted was measured;
#                                               do not change it without re-running
#   @not-implemented                            SECTION F ONLY — describes intent,
#                                               never behaviour. See Section F.
#
#   `@wip` and `@verify-build` are gone. Everything unimplemented now lives in
#   Section F under one tag, so a reader cannot meet it by accident.
#
# CITATION NOTE
#   Inline `# ref:` comments survive only where the cited work still explains a
#   design that exists. Citations attached to deleted behaviour (Zep bi-temporal
#   querying, Leiden/CPM, node2vec parameter tuning, Lost-in-the-Middle payload
#   ordering, HippoRAG routing) were removed, or moved into Section F alongside
#   the behaviour they justify.
#
# VERSION NOTE
#   ArcadeDB 26.5.1 (pinned image). Vector indexes are `LSM_VECTOR` with
#   `METADATA {dimensions:<d>, distanceFunction:'COSINE'}` — there is no
#   quantization option in use and no `vector.fuse`; fusion happens in the
#   service. `<d>` is the embedder's dimension, substituted into the DDL by
#   `ensure_schema(dims)`; the default embedder is `BAAI/bge-small-en-v1.5` at
#   **384** dimensions, in-process, no API key. Graph algorithms (`CALL algo.*`)
#   are reachable ONLY through `language="cypher"` and only after the OLAP
#   analytical view has been created and rebuilt via SQL.
# =============================================================================

@memory-pipeline
Feature: Ontology- and frame-driven agentic memory

  As the GraphKnows memory service
  I want to extract typed entities, ontology-annotated relations and semantic
  frames from conversation and text, consolidate them into one graph per
  namespace, and retrieve through several independent signals fused by rank
  So that an LLM receives grounded, attributable context

  Background:
    Given a running ArcadeDB instance reachable over SQL and openCypher
    And an embedder producing fixed-dimension vectors, local by default
    And an entity ontology whose class definitions are embedded
    And a relation profile whose properties declare domain and range
    And a FrameNet frame-definition embedding index
    And a zero-shot span-and-relation extractor conditioned per chunk

  # ===========================================================================
  # SECTION A — STORAGE & SCHEMA
  # ===========================================================================

  @schema @arcadedb
  Rule: A namespace is one physical database, and STM/LTM is a property on it

    Isolation between tenants is physical: a namespace maps to its own ArcadeDB
    database, so a query in one cannot reach another's records. Short-term and
    long-term memory are NOT separate databases — they are a `state` lifecycle
    on one graph, flipped in place.

    Scenario Outline: A namespace resolves to exactly one database name
      When the database name for namespace "<namespace>" is resolved
      Then it is "<database>"

      Examples: namespace to database
        | namespace | database   |
        |           | mem        |
        | acme      | mem_acme   |
        | ACME      | mem_acme   |
        | a.c-me    | mem_a_c_me |

    Scenario: An over-long namespace keeps distinct inputs distinct
      Given a namespace longer than 48 characters
      When the database name is resolved
      Then the tail is replaced with a short hash of the whole namespace
      And two different over-long namespaces resolve to two different databases

    Scenario: Freshly ingested records are raw and a flush makes them consolidated
      Given a session whose chunks have been ingested but not flushed
      Then every one of its chunks carries state "raw"
      When the session is consolidated
      Then those same chunk records carry state "consolidated"
      And no chunk was copied into a second database

    Scenario Outline: A retrieval scope is a state filter, not a store
      Given a retrieval scope of "<scope>"
      When the collectors run
      Then chunks are filtered by state "<state>"

      Examples: scope to state
        | scope | state        |
        | stm   | raw          |
        | ltm   | consolidated |
        | both  | no filter    |

  @schema @arcadedb @sql
  Rule: The golden-layer schema is created idempotently before any write

    ArcadeDB records are single-typed, so there is no label hierarchy and no
    abstract supertypes: the schema is a flat set of vertex and edge types
    created with IF NOT EXISTS, in order — types, then properties, then indexes.

    Scenario: The vertex types exist
      When the schema is ensured
      Then the following vertex types exist
        | type           | purpose                                           |
        | SESSION        | one ingestion session                             |
        | TURN           | the raw short-term conversation buffer            |
        | FILE           | the source a chunk was parsed from                |
        | CHUNK          | the retrievable passage                           |
        | ENTITY         | a named thing, merged on its normalised name      |
        | TEMPORAL       | one global node per resolved ISO date             |
        | TOPIC          | a community of entities                           |
        | FRAME          | one FrameNet frame, canonical for the namespace   |
        | FRAME_INSTANCE | a role-filled reading of an evoked frame          |
        | ONTOLOGY       | the loaded ontology's identity and term count     |
        | ONTOLOGY_CLASS | one class, with its definition and embedding      |
      And there is no ONTOLOGY_PROPERTY vertex type
      # Properties are never persisted: domain/range validation reads the
      # in-memory term from the loader, so property vertices would be write-only.

    Scenario: The edge types exist
      When the schema is ensured
      Then the following edge types exist
        | edge            | connects                 |
        | PART_OF         | CHUNK -> FILE            |
        | IN_SESSION      | FILE or TURN -> SESSION  |
        | MENTIONS        | CHUNK -> ENTITY          |
        | MENTIONS_DATE   | CHUNK -> TEMPORAL        |
        | HAS_TOPIC       | CHUNK -> TOPIC           |
        | IN_TOPIC        | ENTITY -> TOPIC          |
        | EVOKED_BY       | FRAME -> CHUNK           |
        | FRAME_EVOKED_IN | FRAME_INSTANCE -> CHUNK  |
        | PLAYS_ROLE      | FRAME_INSTANCE -> ENTITY |
        | REL             | ENTITY -> ENTITY         |
        | INSTANCE_OF     | ENTITY -> ONTOLOGY_CLASS |
        | MATCHES_CLASS   | CHUNK -> ONTOLOGY_CLASS  |
        | BROADER         | FRAME -> FRAME, and      |
        |                 | ONTOLOGY_CLASS -> CLASS  |
      # ONE edge type carries both subsumption hierarchies, because both are
      # read identically: one hop DOWN from a query match, at a discount, to
      # reach the specific thing a general query asked about.

    Scenario: Every semantic fact is one REL edge whose predicate is a property
      Given relations with the predicates "worksFor", "knows" and "find"
      When they are written
      Then all three are edges of the single type "REL"
      And each records its predicate in the "predicate" property
      And no edge type was created per predicate
      # A type-per-predicate minted hundreds of near-empty types plus a DDL
      # round-trip on the ingest hot path, and reserved words had to be escaped
      # (find -> REL_FIND) — so the type name was never authoritative anyway.

    Scenario: A REL edge carries its provenance, annotation and validity fields
      When a relation edge is inspected
      Then it declares the following properties
        | property          | meaning                                            |
        | predicate         | the extracted predicate label                      |
        | fact              | the evidence sentence                              |
        | evidence          | the list of chunk ids that asserted it             |
        | confidence        | highest score any assertion gave it                |
        | source            | which extractor produced it                        |
        | ontology_property | the ontology property it mapped onto, or null      |
        | ontology_blocked  | sticky refusal by the domain or range gate         |
        | verifier_score    | the semantic verifier's score, or null if unscored |
        | valid_at          | world-time validity start, when a caller supplied  |
        | invalid_at        | set when superseded; the edge is never deleted     |
        | is_current        | false once superseded                              |
        | state             | raw or consolidated                                |
        | created_at        | when the system first learned it                   |
      And an unscored edge leaves verifier_score NULL rather than writing 0.0

    Scenario: Identity is enforced by unique indexes
      When the schema is ensured
      Then the following unique indexes exist
        | type           | property   |
        | SESSION        | session_id |
        | TURN           | turn_id    |
        | FILE           | file_id    |
        | CHUNK          | chunk_id   |
        | ENTITY         | name_norm  |
        | TEMPORAL       | date       |
        | TOPIC          | topic_id   |
        | FRAME          | id         |
        | FRAME_INSTANCE | id         |
        | ONTOLOGY_CLASS | uri        |
      And ENTITY.name_norm is the casefolded, whitespace-collapsed name

    Scenario: A vector index exists for every embedded vertex
      When the schema is ensured with the embedder's dimension
      Then an LSM_VECTOR index with distanceFunction COSINE exists on each of
        | property                 |
        | CHUNK.embedding          |
        | TOPIC.embedding          |
        | FRAME.embedding          |
        | ONTOLOGY_CLASS.embedding |
      And each index declares the dimension the running embedder produces

    @doc
    Scenario: The vector index DDL is recorded for review
      Then the index statement has the shape
        """sql
        CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR
          METADATA {dimensions:384, distanceFunction:'COSINE'}
        """

    Scenario: There is no full-text index; lexical matching is a scan
      When the lexical collector runs
      Then chunks are found by a CONTAINS scan over search_text, falling back to text
      And the candidates are re-ranked in memory by BM25
      And no SEARCH_INDEX call is issued
      # A FULL_TEXT index is a later phase. The previous revision of this file
      # asserted one existed over Entity name/aliases/description. It does not.

  @schema @arcadedb
  Rule: Sessions are a filter, not a partition

    Scenario: Chunks from two sessions live in one database
      Given session "s-001" and session "s-002" in namespace "acme"
      When both are ingested
      Then both sessions' chunks are records of the same CHUNK type
      And a collector scoped to "s-001" filters on session_id

    Scenario: Two namespaces cannot see each other at all
      Given namespace "acme" and namespace "globex"
      When a query runs against "acme"
      Then it addresses the database "mem_acme"
      And no filter is needed to exclude "globex", which is a different database

  # ===========================================================================
  # SECTION B — VOCABULARIES
  # ===========================================================================

  @ontology @schema
  Rule: One ontology, lexicalized as a SKOS scheme, plus an input overlay

    The bundled vocabulary is the Common Core Ontologies, and it serves both
    jobs. The previous split — schema.org for entity types, a hand-written
    profile for relations — is gone.

    CCO alone is not sufficient, and the reason is structural rather than a
    tuning failure: BFO models everyday assertions as PROCESSES WITH
    PARTICIPANTS, so CCO has no binary predicate for "works for" or "plans to".
    An INPUT SKOS OVERLAY supplies exactly those, as data.

    Scenario: The bundled ontology is the CCO digest
      When the entity ontology is loaded with its default source
      Then it is "assets/cco/cco.json"
      And it yields 1437 classes and 264 object properties
      And 234 of those properties declare both a domain and a range
      And its classes declare subClassOf parents reaching BFO
      # test: tests/ontology/test_cco_digest.py

    Scenario: The relation vocabulary defaults to the same digest
      When the relation ontology is loaded with its default source
      Then it is the same file as the entity ontology
      And the setting remains separate, so a deployment can still split them
      # test: tests/config/test_mode_presets.py

    Scenario: CCO types its relations against BFO, which is an import
      Given the digest builder is given the CCO modules alone
      Then 199 of 224 properties resolve no domain or range
      And the domain/range gate can therefore offer almost nothing
      When "cco-imports/bfo-core.ttl" is parsed alongside
      Then those constraints resolve and typed properties rise to 234
      # BFO is where CCO's relation types actually live; omitting the import
      # empties the relation vocabulary silently.
      # test: tests/ontology/test_cco_digest.py::test_a_bfo_typed_constraint_resolves_to_a_label

    Scenario: The subclass chain reaches BFO, so subsumption is usable
      When a Person's ancestors are computed
      Then the chain is Animal, Organism, object, material entity,
        independent continuant, continuant, entity
      And a property declared over any of those is satisfiable by a Person
      # Under the old schema.org digest the chain degenerated to the class
      # itself, so every gate reading it was inert.
      # test: tests/ontology/test_cco_digest.py::test_the_subclass_chain_reaches_bfo

    @measured
    Scenario: The ancestor walk is breadth-first and case-insensitive at entry
      Given the scheme is keyed by prefLabel "Person"
      When ancestors are requested for the extracted type "person"
      Then the same chain is returned as for "Person"
      And every declared parent is walked, not just the first
      And a concept is never its own ancestor, even in a cyclic hierarchy
      # Both halves were live bugs. The scheme is keyed by prefLabel while every
      # caller hands over the COARSE types entity_labels emits ("person"), so an
      # exact lookup missed all of them and the domain/range widening this
      # function exists for contributed NOTHING: the satisfied set was the 7
      # coarse labels, now 16. Following broader[0] alone also dropped whole
      # branches of CCO's multiple inheritance.
      # test: tests/ontology/test_skos_lexicalization.py::TestAncestors

    @measured
    Scenario: Upper-ontology plumbing is excluded from the STRUCTURAL offer
      Given the widened satisfaction set above
      When relation labels are selected for a chunk that speaks none of them
      Then "continuant part of", "is predecessor of" and "member part of"
        are NOT offered
      And a property whose own name the text SPEAKS is still offered
      # A BFO-typed property is satisfiable by literally anything, because the
      # ancestor walk climbs every entity into material entity / continuant. It
      # would win budget slots on every chunk while asserting a relation between
      # categories of being that no speaker uttered — the permits/requires
      # failure that produced 64 of 286 REL edges. The LEXICAL half keeps them:
      # text that says "requires" IS asserting the property.
      # test: tests/ingestion/test_lexical_labels.py::TestDomainRangeSatisfaction

    Scenario: The input overlay supplies what BFO does not model as a predicate
      When the SKOS scheme is built with its default overlay
      Then the overlay is "assets/cco/conversational.json"
      And it contributes properties including "works for", "plans to", "likes",
        "attended", "teaches" and "formerly worked for"
      And each carries authored altLabels, a broader link into CCO, and
        a domain and range written in CCO's own class vocabulary
      And setting GRAPHKNOWS_ONTOLOGY_OVERLAY to "-" disables it
      # Data, not code: swapping the file swaps the vocabulary.
      # test: tests/ingestion/test_lexical_labels.py::TestOverlay

    Scenario: Authored altLabels are lemmatised on load
      Given an overlay property authored with the altLabel "lost"
      When the scheme is built
      Then both "lost" and "lose" are altLabels of that concept
      # Matching is on spaCy lemmas, so a surface form authored in the past
      # tense matches NOTHING. This read as a model failure and was a string
      # normalisation bug.
      # test: tests/ingestion/test_lexical_labels.py::test_authored_alt_labels_are_lemmatised

    Scenario: The digest is pre-parsed JSON and needs no rdflib
      When the ontology is loaded
      Then the digest shape is recognised before any RDF parser is tried
      And no rdflib import occurs
      # rdflib ships only in the `assisted` extra, so the default llm_free
      # install must reach its ontology through the standard library alone.
      # This is also what makes both memory modes load it identically.
      # test: tests/ontology/test_cco_digest.py::test_load_does_not_import_rdflib

    Scenario: Ontology injection is always on
      Given no ontology environment variable is set
      When the ontology source is resolved
      Then it resolves to the bundled digest
      And an empty environment value also resolves to the bundled digest
      And there is no configuration that turns ontology injection off

  @ontology @model
  Rule: Definition embeddings survive only as the pre-lexical fallback

    Per-chunk label selection is LEXICAL (see Section C). Definition embeddings
    remain for the chunk MATCHES_CLASS tags, the entity INSTANCE_OF mapping and
    the optional ontology retrieval channel — never for choosing which labels a
    chunk is offered.

    @measured
    Scenario: Embedding a chunk against class definitions does not discriminate
      Given chunk-to-class matching by cosine over the class definitions
      When 40 real conversation turns are tagged
      Then only 11% of the tags share even one content word with their chunk
      And "Lost my job as a banker" tags "Grocery Store"
      And the three most common tags corpus-wide are
        "Act of Thanking", "Grocery Store" and "Shop"
      # Four independent attempts failed the same way: content-word queries,
      # label-only indexes, a lexical gate, and entity-mention pooling. The
      # registers differ — casual speech against genus-differentia prose — and
      # no ranking fixes that. Do not re-try it for LABEL SELECTION.

    Scenario: Class embeddings are built from label and definition
      When the ontology index is built
      Then each class is embedded as its label followed by its definition
      And the matrix is cached to disk keyed by embed model, dimension and encoding
      And a change to the way text is encoded invalidates that cache

    Scenario: A class with no definition indexes on its label alone
      Given an ontology class with a label and no definition
      When the index is built
      Then the class is still matchable, from its label

  @frames @model
  Rule: The FrameNet index embeds every frame definition once

    Scenario: Frame definitions are embedded and cached
      When the frame index is built
      Then every FrameNet frame is embedded as its name followed by its definition
      And the L2-normalised matrix is cached to disk with its frame-name list
      And the cache key includes the embed model and dimension

    Scenario: Frame matching is top-k above a permissive floor
      Given a pre-computed embedding
      When frames are matched
      Then the top 5 frames by cosine are returned
      And only those at or above a similarity floor of 0.20 are kept
      # Absolute cosines run low under the configured embedders (~0.30-0.39),
      # so the floor is permissive and the ranking does the work.

    Scenario: Frame sub-ontologies are read lazily from nltk
      Given a frame a chunk actually evoked
      When its sub-ontology is requested
      Then its definition, core frame elements, peripheral frame elements and
        lexical units are read from nltk FrameNet and cached
      And frames no chunk evoked are never materialised

  # ===========================================================================
  # SECTION C — INGESTION
  # ===========================================================================

  @ingest
  Rule: Text is chunked along headings, turns are chunked by sliding window

    Scenario: Markdown is chunked by heading with small sections merged
      Given a Markdown document with headings
      When it is parsed
      Then each chunk carries its heading breadcrumb path and its ordinal position
      And sections too small to stand alone are merged until the token budget would overflow
      And no chunk exceeds the configured token budget
      # test: tests/ingestion/test_parsers.py

    Scenario: Plain text gets one synthetic heading which is then stripped
      Given plain text with no headings
      When it is parsed
      Then it is treated as one implicit section headed "## Content"
      And that synthetic heading is stripped before embedding and before display

    Scenario: A chunk identifier is derived from its content and position
      When a document is chunked twice
      Then each chunk's identifier is the same both times
      And the identifier is a hash of session, source, character offset and chunk text
      # The text is in the key because a character offset alone collides when one
      # section yields several chunks that resolve to the same offset.

    Scenario: Conversation turns become one chunk per 5-turn window
      When turns are ingested with the default settings
      Then a CHUNK of kind "turn" is written every 3 turns, covering the last 5
      And 369 turns yield 123 chunks
      And the last turn of a batch forces the pending window out, so no tail is lost
      And each chunk records the speaker of its final turn, not merely the coarse role
      # A lone turn is a poor retrieval unit: "She said yes" carries nothing an
      # embedding can match, and the turn that named her is a different chunk.
      # The extractor has read a 5-turn window since `extract_window` landed;
      # this makes the STORED, embedded, retrieved unit that same window.
      # Extraction therefore fires once per window rather than once per turn
      # over substantially the same text, so ingest does ~3x less work.
      # test: tests/memory/stm/test_service_messages.py::TestWindowChunking

    Scenario: A multi-turn window is speaker-prefixed but never deixis-resolved
      Given a window covering turns by Gina and Jon
      When the chunk is written
      Then each line reads "Speaker: utterance", because one text has many speakers
      And the text is NOT deixis-resolved
      And that same prefixed text is what gets embedded
      # The extractor benefits from "Gina lost Gina's job"; a reader quoting the
      # memory back does not.

    Scenario: A window of one turn is still stored verbatim
      Given the session's first turn, with no prior turns to join
      When the chunk is written
      Then CHUNK.text is that turn byte for byte, with no speaker prefix
      And the embedding is still computed from the attributed search_surface
      # Two separate measured contracts: storage stays verbatim, the EMBEDDING
      # carries the attribution the utterance itself omits.

    Scenario: The provenance filter is skipped when the chunk is the window
      When entities are extracted for a window chunk
      Then spans from every turn in the window become this chunk's entities
      # The offset filter exists to stop a TURN-chunk re-minting the previous
      # four turns' entities. When the chunk IS those turns, it would drop all
      # but the last turn's.

    Scenario: Window chunking has an exact control arm
      Given GRAPHKNOWS_WINDOW_CHUNKS is false
      When turns are ingested
      Then one chunk is written per turn, exactly as before
      And its identifier is a hash of session, file, sequence and cleaned text
      # An exact replay, which is what makes it an A/B control rather than an
      # approximation of one.
      # test: tests/memory/stm/test_service_messages.py::test_window_chunking_off_restores_one_chunk_per_turn

    @measured
    Scenario: The window is what makes the chunk worth retrieving
      Given conv-30 fed turn by turn, 81 questions, --reset on both arms
      When chunks are single turns
      Then evidence_recall@probe is 0.795
      When chunks are 5-turn windows
      Then evidence_recall@probe is 0.976
      # +0.181 exact, not judge-scored. See also the ts scenario in Section E:
      # retrieval and generation were two separate wins.

    Scenario: An empty turn is dropped loudly
      When a turn with no non-whitespace text is ingested
      Then no chunk is written
      And the drop is logged naming the session, sequence and role
      # "200 turns in, 0 chunks out" must not read the same as a wedged extractor.

  @ingest
  Rule: The retrieval surface carries the speaker; the stored text does not

    An utterance usually omits its own speaker. "I'm opening a dance studio" is
    a fact about Gina and names her nowhere, so neither its embedding nor a
    lexical scan of it can be reached by a question about Gina.

    Scenario: The embedded and scanned surface is speaker-prefixed
      Given a turn spoken by "Gina" reading "I'm opening a dance studio"
      When the turn is ingested
      Then CHUNK.search_text is "Gina: I'm opening a dance studio"
      And CHUNK.text is "I'm opening a dance studio", verbatim
      And the embedding is computed from search_text

    Scenario: A speaker already present in the text is not repeated
      Given a turn whose text already opens with the speaker's name and a colon
      When the retrieval surface is built
      Then the text is returned unchanged

    Scenario: The extractor still sees the bare text
      When entities are extracted from a turn
      Then the extractor input is the cleaned text without the speaker prefix
      # Putting the name in the extractor's input is what minted speakers as
      # ENTITY nodes and saturated the entity graph.

  @ingest @extract @model
  Rule: Extraction is two-pass, because the relation vocabulary comes from pass 1

    The relation properties offered to the extractor are selected from the
    ENTITY TYPES the text contains, so they cannot be known before entities have
    been extracted. The second forward pass is the price of selecting
    structurally rather than topically.

    Scenario: Pass 1 extracts entities with no relation labels
      When a chunk is extracted
      Then the first pass is issued with a null relation label set
      And the extractor returns entity spans only
      And the null case is logged at debug, not warned about
      # `None` (not asked) and `{}` (asked, matched nothing) are different facts.
      # Warning on `None` fires once per chunk by design and drowns the real case.

    Scenario: Pass 2 re-extracts with the properties pass 1's types evoked
      Given pass 1 found entities typed Person and Organization
      When the relation label set is selected
      Then the chunk is re-extracted with those property labels as the relation spec
      And the relations of the second result replace the first result's

    Scenario: A chunk whose types evoke no property is not re-extracted
      Given a chunk whose entity types map to nothing in the relation profile
      When the relation pass runs
      Then that chunk keeps its pass-1 result
      And the skip is logged with how many chunks it covered

    Scenario: The batch path pays the second pass only where it can produce something
      Given a batch of 100 chunks of which 30 evoke a property
      When the relation pass runs
      Then 30 chunks are re-extracted and 70 are not
      And the log reports 30 of 100 with the elapsed time per chunk

  @extract @ingest @measured
  Rule: The extraction window is overlapping, per-turn attributed, and guarded

    One turn read alone loses what the turns before it established. The window
    widens what the EXTRACTOR sees — and, since `window_chunks` became the
    default, it is also what is stored, embedded and retrieved (see "turns are
    chunked by sliding window" above). The deixis-resolved form below is the
    extractor's input only; storage keeps the verbatim lines.

    Scenario: Windows overlap so no turn boundary is a blind spot
      When turns are ingested with the default settings
      Then each window is 5 turns and advances by 3
      And every adjacent pair of turns co-occurs in some window
      # A relation whose endpoints straddle a boundary is invisible to every
      # window. Measured at equal turn coverage: 2 turns of overlap yields +35%
      # distinct edges. The duplicates cost nothing, because write_relation
      # MERGEs on (head, predicate, tail) and accumulates evidence.

    Scenario: Deixis is resolved PER TURN before the lines are joined
      Given a window in which Jon says "Lost my job as a banker"
        and Gina says "I also lost my job at Door Dash"
      When the window text is built
      Then Jon's line reads "Lost Jon's job as a banker"
      And Gina's line reads "Gina also lost Gina's job at Door Dash"
      # A window has many speakers and ONE text. Left unresolved, the extractor
      # attributed Gina's job to Jon.
      # test: tests/ingestion/test_lexical_labels.py::TestDeixis

    Scenario: Second person resolves to the addressee, and only when unambiguous
      Given a two-party window
      Then "your" resolves to the other participant
      Given a window with three or more speakers
      Then second person is left untouched
      # Above two participants the addressee is a guess, and a wrong name is
      # worse than a pronoun.

    Scenario: Contractions are expanded before the pronoun rules run
      Given a turn by "Gina" reading "I'm starting a studio"
      When the window text is built
      Then it reads "Gina is starting a studio"
      And no name is ever emitted glued to a contraction tail
      # An apostrophe is a word boundary, so a bare pronoun rule rewrote "I'm"
      # to "Gina'm" — a SECOND person node that took all of that speaker's
      # relations with it.
      # test: tests/ingestion/test_lexical_labels.py::TestContractions

    Scenario: Only the current turn's spans become this chunk's entities
      Given GRAPHKNOWS_WINDOW_CHUNKS is false, so a chunk is one turn
      When entities are extracted from a window
      Then spans lying before the current turn's offset are discarded
      And relations are NOT filtered that way
      # Earlier turns already minted their entities against their own chunks;
      # a cross-turn TRIPLE is the entire point of the window. Under the default
      # (the chunk IS the window) this filter is skipped — every entity in the
      # window really is mentioned by that chunk.

    Scenario: A relation's subject must be supported by the asserting turn
      Given a window where two people each report losing a job
      When relations are extracted
      Then only the nearest person named in the object's own turn may head it
      # Deixis fixes pronouns, not crossed subjects. Presence in the turn is not
      # enough — deixis itself writes the addressee's name into the line — so
      # the guard keys on PROXIMITY.
      # test: tests/ingestion/test_lexical_labels.py::TestAttributionGuard

  @ingest @extract @model
  Rule: First-person deixis is resolved for the relation pass only

    A relation needs two entity SPANS, and "I" / "my" / "I've" are not spans —
    so in dialogue, where speakers state facts about themselves in the first
    person, the subject of nearly every stated fact is invisible.

    @measured
    Scenario: Resolving deixis is what makes conversational relations exist
      Given 40 real conversation turns, one model, one label set, one threshold
      When relations are extracted from the raw text
      Then 3 relations are produced
      When relations are extracted from the speaker-resolved surface
      Then 19 relations are produced

    Scenario: The surface is rebuilt token-wise, not by regex
      Given a turn by "Gina" reading "I'm over the moon"
      When the surface is built
      Then it reads "Gina is over the moon"
      And the clitic is expanded to a third-person form rather than left as "Gina'm"
      # A regex over the raw text mangles contractions: it produced
      # "gina is over the moon be".
      # test: tests/extraction/test_lingfeatures.py

    Scenario: A genitive pronoun keeps its genitive marker
      Given a turn by "Gina" reading "My collection has twenty pieces"
      When the surface is built
      Then it reads "Gina's collection has twenty pieces"

    Scenario: Each line of a multi-turn chunk resolves to its own speaker
      Given a chunk of several "timestamp | Speaker: text" lines
      When the surface is built
      Then a first-person pronoun on each line resolves to that line's speaker
      And a line with no turn prefix falls back to the caller-supplied speaker

    Scenario: Second person resolves only when the chunk has exactly two speakers
      Given a chunk with exactly two speakers
      When the surface is built
      Then "you" resolves to the other of the two
      Given a chunk with one speaker
      When the surface is built
      Then "you" is left alone, and the pronoun guard later drops any relation on it

    Scenario: Entities and offsets are unaffected by deixis resolution
      Given a chunk carrying first-person deixis and a non-empty relation spec
      When extraction runs
      Then entities are inferred from the ORIGINAL text
      And relations are inferred from the resolved surface
      And character offsets, alias candidates and overlap resolution anchor on the original
      And CHUNK.text is stored verbatim
      # Head and tail are matched downstream by NAME, not by offset, which is
      # what lets the two passes read different strings.

    Scenario: No known speaker means no surface and a raw-text relation pass
      Given a chunk in which no speaker can be parsed or supplied
      When the surface is built
      Then it is empty
      And relations are read off the raw text, which is logged at debug

    Scenario: Text with no deixis round-trips verbatim
      When a chunk containing no first- or second-person pronoun is resolved
      Then the surface equals the original text and is treated as absent

    Scenario: Main-verb agreement is deliberately left crude
      Given a turn by "Gina" reading "I work at the store"
      When the surface is built
      Then it reads "Gina work at the store"
      # Inflecting the verb needs a morphologiser, and the recall gain above was
      # measured without one. Only the clitic — which is not a word at all after
      # the swap — is repaired.

  @ingest @extract @model
  Rule: Entity labels are unioned; relation labels are the whole spec

    The extractor is zero-shot: it can only emit a type it was handed.

    Scenario: Three label sources are unioned for entities
      When the entity label list is built for a chunk
      Then it is the chunk's own spaCy-observed labels,
        then the ontology class hints,
        then the base inventory,
        de-duplicated in that order
      # The ontology hints are LEXICALLY selected when a parser is available —
      # see "Labels are selected by lexical evidence" below. Without one the
      # embedding hints are used and the behaviour is unchanged.

    @measured
    Scenario: An empty spaCy observation is not an empty schema
      Given a chunk in which spaCy observes no named entity
      When extraction runs
      Then the chunk is still extracted from, using the base inventory
      # Skipping such chunks was a cost optimisation that assumed spaCy sees what
      # matters. Measured on a 369-turn conversation, 170 turns (46%) observed no
      # label and were never extracted from, including plain factual statements.
      # test: tests/extraction/test_blind_ner_fallback.py

    @measured
    Scenario: The base inventory names things, never states or processes
      When the base entity inventory is inspected
      Then it contains "person", "organization", "location", "occupation", "animal", "food"
      And it does NOT contain "activity"
      And it does NOT contain "emotion"
      # Asked for an "activity" a zero-shot extractor returns every verb, and for
      # an "emotion" every adjective. Measured: those two labels produced 906 of
      # 2105 entity nodes (43%), the top "entities" in the graph being `keep` x56,
      # `going` x32, `make` x29 — and they reached the generator's fact-sheet as
      # frame role fillers. No downstream name filter repairs a label that tells
      # the model to tag the wrong part of speech; the fix is not asking for it.

    Scenario: An empty relation spec means the extractor is asked for entities only
      Given no ontology property matched a chunk
      When extraction runs
      Then the relation label list is empty
      And the extractor is asked for entities only
      And a warning names the chunk length and its opening characters
      # An unlogged empty result is indistinguishable from a broken extractor.

    Scenario: The extractor is invoked once per whole chunk
      When a chunk is extracted
      Then there is one forward pass over the whole text, not sub-windows
      # The old 128-token windows cost ~4 model calls per chunk and existed
      # because the previous model attended poorly to long spans.

  @extract @ontology @model
  Rule: Labels are selected by lexical evidence over a SKOS scheme

    Selection asks which concepts the chunk's WORDS evoke, matching spaCy lemmas
    against altLabels. Nothing is embedded. This replaced definition-embedding
    selection, which could not discriminate (see Section B).

    Scenario: A concept is evoked by a lemma, part-of-speech gated
      Given a turn containing "I lost my job at Door Dash"
      When the relation labels are selected
      Then "formerly worked for" is among them
      And a class concept may only be evoked by a NOUN
      And a property concept may be evoked by a NOUN or a VERB
      # CCO names relations as relational nouns ("has brother"), so a verb-only
      # gate would never fire on the cases this corpus is full of.
      # test: tests/ingestion/test_lexical_labels.py::TestLexicalSelection

    @measured
    Scenario: A multi-word label needs every content word present
      Given a turn containing "take a shot at" and "this month"
      When class labels are selected
      Then "Round Shot" is NOT evoked
      And "Calendar Month" is NOT evoked
      # A single token of a multi-word label is not evidence for the term. The
      # document-frequency filter cannot catch these: "shot" is genuinely rare.

    @measured
    Scenario: Ubiquitous lemmas are dropped, but only once there is text to judge
      When the lexical index is built over a corpus
      Then a lemma appearing in more than 15% of documents is dropped
      And that rule applies only once the lemma appears in at least 3 documents
      # Without the absolute floor a 5-document corpus discards every content
      # word — one occurrence is already 20% — and the vocabulary empties. The
      # document frequency is measured over TURNS, never over windows: windows
      # overlap, so every word looks ubiquitous.

    @measured
    Scenario: The offer stays small because the score depends on its size
      When labels are offered to the extractor
      Then at most 14 entity labels and 10 relation labels are offered
      # Measured: the same correct triple scores 0.98 with 4 labels offered and
      # 0.67 with 39. relex's confidence is RELATIVE to the offered set, so a
      # selector that returns everything it matched destroys the signal it feeds.

    Scenario: The coarse types are always offered
      When entity labels are selected for any chunk
      Then "person", "organization" and "facility" are among them
      # They are the measured-best types AND are CCO classes verbatim, so they
      # cost nothing ontologically and the offer is never empty.

    Scenario: Selection degrades rather than fails without a parser
      Given an extractor exposing no spaCy pipeline
      When labels are selected
      Then lexical selection is switched off and the fact is logged
      And the previous embedding-hint behaviour is used

  @ingest @extract @model
  Rule: Relation properties are selected by domain and range, not by cosine

    The ontology analogue of the FrameNet layer: a frame is evoked by a trigger
    in the text and its core elements then define the schema to fill; here the
    extracted entity types evoke the properties whose declared domain and range
    they satisfy. Anchored in what the text contains, not what it is near.

    @measured
    Scenario: Whole-chunk cosine selection was measured and produced nothing
      Given properties selected by cosine between the chunk embedding and 945 properties
      When 60 real conversation turns are ingested
      Then 0 relations are produced
      And the wrong terms outscore the right ones with no floor separating them
      # "Still following my passion for dance" ranks `liveBlogUpdate` at 0.64
      # above every property it actually asserts. Do not re-try this.

    Scenario: Both endpoints must be present for a property to be offered
      Given a chunk containing one Person and no Organization
      When the relation label set is selected
      Then "knows" and "likes" are offered
      And "worksFor" is not offered
      # An extractor offered `worksFor` with no Organization present could only
      # invent the other end.

    Scenario: An entity type is mapped to a profile class before selection
      Given the extracted type "PERSON"
      When it is mapped to the relation profile
      Then it maps to "Person" with score 1.0 by exact label match
      Given the extracted type "noun.person"
      When it is mapped
      Then it is embedded and matched against the class definitions above a floor of 0.45
      And a type that maps to nothing contributes no property to the offer

    Scenario: Class ancestors widen the match
      Given a type mapped to "Country"
      When the present class set is computed
      Then it contains "Country" and its transitive subClassOf parents

    Scenario: The offer is capped at 15 and allocated round-robin
      Given a chunk whose types evoke 40 properties across several domain-range pairs
      When the relation label set is selected
      Then at most 15 properties are offered
      And they are interleaved across the domain-range groups, not taken in profile order
      # Nine interpersonal Person->Person properties sit at the top of the
      # profile and would otherwise take every slot the moment a person is
      # mentioned, deleting the event, place and employment properties entirely.

    @measured
    Scenario: The cap is a measured band, not a free parameter
      Given 5 properties offered on two friends chatting
      Then the extractor emits "Gina spouse Jon" and "Gina children Jon", neither true
      Given about 14 properties offered on the same turns
      Then the extractor emits colleague, owns and follows
      # A closed vocabulary with no way to decline FORCES a pick. The extractor
      # families in use are trained at <= 25 labels per instance, so the ceiling
      # is real too. 15 sits in the measured band.

    Scenario: An empty offer is a logged outcome
      Given a chunk whose extracted types evoke no property
      When the relation label set is selected
      Then it is empty
      And the log names the extracted types and what they mapped to
      # test: tests/extraction/test_ontology_relation_labels.py

    Scenario: An unloadable relation profile degrades loudly
      Given the bundled relation profile will not load
      When the ingest handler is built
      Then it falls back to the entity ontology's properties
      And an error line says so, naming the source
      # Silence here reads exactly like "no relations in this corpus", which is
      # the defect this layer keeps having.

  @ingest @extract @model
  Rule: Overlapping entity spans resolve to one entity, and the discards become aliases

    Nested spans are the extractor's DESIGN — it runs with flat_ner disabled —
    not a defect. "Customers love new offers and promotions." comes back with
    'new', 'new offers' and 'offers' all typed the same: three nodes, one thing.

    Scenario: The longer span wins and confidence breaks the tie
      Given "Ada Lovelace" at confidence 0.9 and "Lovelace" at confidence 0.7, overlapping
      When overlaps are resolved
      Then one entity remains, named "Ada Lovelace"
      And "Lovelace" is recorded as an alias candidate
      # test: tests/ingestion/test_overlapping_spans.py

    Scenario: Type is deliberately not consulted
      Given two spans covering the same characters typed work_of_art and person
      When overlaps are resolved
      Then they collapse to one entity
      # Two spans over the same characters are one mention of one thing, whatever
      # labels the model hung on them.

    Scenario: Overlap is decided on real character offsets
      Given two spans with offsets that do not intersect
      When overlaps are resolved
      Then both are retained
      Given spans with no offsets at all
      When overlaps are resolved
      Then all are retained, because a missing offset is not evidence of overlap

    Scenario: A trimmed surface is re-anchored on what survived
      Given the raw span "Caroline: Thanks, Melanie" reduced to "Caroline"
      When the span offsets are recomputed
      Then they cover only "Caroline"
      And "Melanie" is not swallowed as an alias of "Caroline"

    Scenario: The same name twice in one chunk is deduplicated on the merge key
      Given one chunk yielding two non-overlapping spans of the same name
      When the entity list is built
      Then one entity survives, carrying the higher score's type
      And the other surface is carried as an alias
      # The graph merges on a UNIQUE name_norm with a batched UNWIND MERGE, so a
      # repeat here raises DuplicatedKeyException and kills the whole document's
      # ingest — silently removing every chunk in it from the graph.

    Scenario: Alias candidates are persisted on the entity
      When the entity is written
      Then its discarded surface forms are stored in ENTITY.aliases
      And an entity with no discarded variant carries no aliases key

  @ingest @extract
  Rule: A span is cleaned, reduced and gated before it can become an entity

    Scenario: Enclosing quotes and trailing punctuation are stripped
      Given the span '"Becoming Nicole"'
      When it is cleaned
      Then it is "Becoming Nicole"
      # The store's merge key derives from this string; ArcadeDB strips quotes
      # from name_norm but not from name, the two diverge, a later MERGE cannot
      # find the record it just wrote, and the write fails on the UNIQUE index —
      # taking the caller's whole document down with it.

    Scenario: A leading article is dropped
      Given the span "a banker"
      When it is cleaned
      Then it is "banker"
      # The determiner forks the merge key into two nodes for one referent, and
      # walked the pair past the same-entity check as "the fair -> fair".

    @measured
    Scenario: A speaker-glued span reduces to the speaker
      Given "Caroline" is a known speaker of this chunk
      And the span "Caroline: Thanks, Melanie"
      When the span is reduced
      Then it is "Caroline"
      # Measured: 27 such names carried 42.5% of all relation edges, fragmenting
      # the real person node. The semantic referent of such a span IS the
      # speaker, and any real entity in the remainder is emitted as its own span.

    Scenario: A speaker name opening a clause reduces to the speaker
      Given "Gina" is a known speaker
      And the span "Gina is over the moon because"
      When the span is reduced
      Then it is "Gina"
      And "Gina Torres" is left alone, because its remainder is capitalised
      # On raw text this shape is a pronoun clause the junk gate drops; the
      # deixis-resolved surface rewrites the pronoun to a NAME and walks past it.

    Scenario: A bare genitive of a speaker is that speaker
      Given "Jon" is a known speaker
      And the span "Jon's"
      When the span is reduced
      Then it is "Jon"
      And "Jon's studio" is left alone
      # The genitive forked the node and hid a self-relation outright:
      # "Jon's -[spouse]-> Jon" walked straight past the head != tail check.

    Scenario: A colon alone never triggers reduction
      Given a chunk with no known speakers
      And the span "Charlotte's Web: A Review"
      When the span is reduced
      Then it is unchanged

    @measured
    Scenario: A span this chunk uses only as a verb is not a thing
      Given a chunk using "speaks" only as a verb
      When entities are built
      Then "speaks" is not an entity
      Given a chunk reading "I work at the store after work"
      Then "work" IS kept, because it appears as a noun somewhere in the chunk
      # Measured: 164 of 912 mentioned entities (18.0%) were verb-headed where
      # they occurred. Tagging the same names in ISOLATION mislabels 67 more
      # (7.3%) that the parser reads as nouns in their own sentence, so a lexicon
      # or a bare-string rule would delete exactly those. Context is the gate.

    Scenario: Dates, times and numbers never become entities
      When "23 July 2023", "10:31 am", "2023" and "next Tuesday" are gated
      Then none of them is admitted to the entity graph
      And "May" and "June" ARE admitted, because they are also person names
      And the date signal is not lost: it lives on the chunk's TEMPORAL link
      # test: tests/ingestion/test_entity_hygiene.py

    Scenario: One junk gate is shared by every minting path
      Then the entity/MENTIONS path, the relation endpoints, the frame role
        fillers and the flush-time resolver all call the same predicate
      # The frame role path was the one that did not: measured, it minted 624
      # entity nodes with no type, no confidence and no mention edge — 30% of the
      # graph — and fed them to the generator ("Buyer=Hey Gina!", "Speaker=Yeah").

  @ingest @extract
  Rule: ENTITY.type is the graph's own taxonomy; endpoint types are the richest label

    Two questions with two answers, deliberately.

    @measured
    Scenario: An ontology class name is not stored as an entity type
      Given the extractor echoes back the ontology class label it was offered
      When ENTITY.type is decided
      Then the span is kept and the type is left empty
      # Censused on a 369-turn ingest: 1256 entities over 41 distinct types
      # against a 763/10 baseline, led by EMOTION 275, ACTIVITY 202,
      # RELATIONSHIP 113 — all class labels of the configured ontology. Any
      # ontology default leaks the same way at a smaller scale; under CCO the
      # residue is BFO plumbing (`permits`, `requires`, `is required by`) at
      # roughly 16% of edges.

    Scenario: An empty type leaves room for a later mention to fill it
      Given ENTITY.type is written first-write-wins
      When a later chunk types the same entity from NER
      Then that label is stored
      # Which is why a wrong type is worse than no type.

    Scenario: A relation endpoint takes the richest label available
      When an endpoint's type is decided
      Then the extractor's own type wins, then the NER label,
        then a WordNet supersense translated into the graph's taxonomy
      And a supersense with no defensible equivalent yields an empty type
      # The endpoint type is a GATE input feeding the domain and range checks,
      # so it wants the richest label there is. It must never reach ENTITY.type
      # in WordNet's vocabulary: two vocabularies in one column make the field
      # unqueryable, and the supersense form also reaches the class matcher as an
      # unknown label where it can only ever score by embedding.

  @ingest @extract
  Rule: Structural relation guards drop what no predicate judgement could save

    The label set is CLOSED, so the extractor is FORCED to pick one of the
    properties it was handed. Nothing here can judge a predicate, but the
    structural checks cost nothing and are exactly what the extra recall from
    deixis resolution makes necessary.

    Scenario Outline: Each guard has a named reason
      When a candidate relation "<case>" is guarded
      Then it is dropped with reason "<reason>"

      Examples: relation guards
        | case                                  | reason                     |
        | an empty head, label or tail          | empty argument             |
        | head and tail are the same entity     | self-relation              |
        | head or tail is a bare pronoun        | pronoun endpoint           |
        | head is a date, clause or backchannel | head is not an entity name |
        | tail is a date, clause or backchannel | tail is not an entity name |

    Scenario: Drops are tallied by reason and logged
      When the guard has dropped candidates
      Then one log line reports the total, the candidate count and the per-reason counts
      # An unlogged swallow makes "the model found nothing" and "the guard is
      # eating everything" indistinguishable from outside, which is how this repo
      # has previously shipped inert layers.

    Scenario: The same fact twice in one chunk is emitted once
      Given the extractor emits the same head, predicate and tail from two spans
      When the relation list is built
      Then one relation survives

    Scenario: A semantic verifier drops confident false positives
      Given the relation verifier is enabled with threshold 0.5
      When "Dancing --DO_WITH--> dance" is scored
      Then it is dropped before the graph merge
      And the extractor's own score could not have caught it, because it IS confident
      And a disabled or unloadable verifier becomes a null object that passes everything through
      # A missing model never kills ingest, but it does become a no-op — which is
      # why the score it did compute is persisted (see REL.verifier_score).
      # test: tests/ingestion/test_verifier_score_persisted.py

  @ingest @ontology @arcadedb
  Rule: The ontology annotates; it never filters

    A filter here would reject every open-vocabulary label, which is why one was
    refused before. An entity that maps to nothing keeps its extracted type; a
    predicate that maps to nothing is still written under its extracted label.

    Scenario: A chunk is tagged with the classes it evokes
      When a chunk is ingested
      Then its top 3 ontology classes above a similarity of 0.60 are written as MATCHES_CLASS edges
      And each edge records the similarity
      # The tag floor is stricter than the hint floor: a hint the extractor finds
      # no span for simply goes unused, whereas a persisted tag is signal
      # retrieval will trust.

    Scenario: Class hints are floored, not merely top-k
      Given the per-chunk class hint floor of 0.50
      When hints are selected for a chunk that says "Thanks, Gina!"
      Then the hint list is empty
      And the empty case is counted and logged
      And the extractor falls back to its base inventory for that chunk
      # Top-k alone was enough while the bundled ontology had 11 classes; against
      # 933 it hands twelve arbitrary labels to a chunk that asserts nothing.
      # test: tests/ontology/test_cco_digest.py

    Scenario: The hint floor bounds volume, not quality
      Then absolute cosines against the class space are not calibrated enough to
        separate a right term from a wrong one
      # Re-measured against CCO over 20 real turns with bge-small: the 1401-class
      # space spans 0.168-0.683, median 0.409, p95 0.493, weakest genuine
      # per-turn top-1 0.497. The hint floor is 0.46 and the persisted-tag floor
      # 0.52. A high cosine is not a correctness signal — a thank-you turn's best
      # class is `Artifact History` at 0.545. Re-census before retuning.
      # This floor governs the MATCHES_CLASS tags, not label selection, which is
      # lexical.

    Scenario: An entity is linked to the class it maps onto
      Given an entity typed "PERSON"
      When the ontology layer runs
      Then an INSTANCE_OF edge to the class "Person" is written with score
        = that entity's extractor confidence x the type-to-class mapping score
      # The mapping score alone is identical for every entity of the type, so it
      # can rank nothing; the per-entity factor is what makes two Persons
      # comparable. Repeated writes keep the strongest mention, not the latest.
      Given an entity type with no exact label match
      Then it is embedded against the class definitions and linked only above 0.45
      And an unmapped entity is written exactly as it would have been without an ontology

    Scenario: Class mapping is by embedding, never by n-gram
      Then single-word type labels are matched against class DEFINITIONS
      # On single-word labels the character-n-gram similarity is dominated by
      # shared suffixes and confidently returns nonsense (LOCATION ->
      # Organization, ANIMAL -> Goal). The PREDICATE mapper does use n-grams
      # plus a surface-alias table, which suits multi-word predicates.

    Scenario: Type-to-class mapping is memoised per distinct type string
      When a corpus with 41 distinct entity types is ingested
      Then at most 41 type embeddings are computed, not one per entity

  @ingest @ontology
  Rule: Domain and range are both validated, and a refusal is sticky

    @measured
    Scenario: The domain gate refuses a non-person subject on a person property
      Given "worksFor" declares the domain "Person"
      And a subject typed LOCATION
      When the domain gate runs
      Then the property mapping is refused
      # The defect shape is "London works_as Analytical Engine": a property that
      # requires a human subject, asserted by a place.
      # test: tests/ontology/test_relation_conformance.py

    Scenario: The domain gate is enforced only where Person is declared
      Given a property whose declared domains do not include Person
      When the domain gate runs
      Then it admits
      # Every other domain miss is the ontology being narrower than the data.

    @measured
    Scenario: The domain gate needs no confidence floor
      Then a subject mapped to any non-person class supports the same verdict
      # This gate concludes "the subject is NOT a person", which every non-person
      # class supports equally: whether GPE landed on Event (0.494) or Place does
      # not change it. Censused against the profile, no type denoting a person
      # maps to a non-person class — PERSON 1.000 and noun.person 0.840 both
      # reach Person, while LOCATION (0.653), noun.location (0.843), OCCUPATION
      # (0.613) and ORGANIZATION (1.000) are all correctly not people.

    @measured
    Scenario: The range gate fires only when the object is confidently a person
      Given "owns" declares the range "Possession"
      And an object typed PERSON at class similarity 1.000
      When the range gate runs
      Then it refuses
      Given an object typed PRODUCT at class similarity 0.580
      When the range gate runs
      Then it admits
      # Censused over every endpoint type in a 369-turn conversation, exactly
      # three map onto Person: PERSON 1.000 and noun.person 0.840 must block,
      # PRODUCT 0.580 must not. The floor sits at 0.7, the midpoint of that gap.
      # An earlier 0.9 was set from the label-hit score alone and sat on the
      # wrong side of it. Retune only against a fresh census; the two neighbours
      # are 0.26 apart.

    @measured
    Scenario: Full range enforcement was measured and is far too strict
      Given 27 "owns" edges in a real conversation
      Then 26 are correct and carry types the narrow declared range does not admit
      And full enforcement destroyed all 26 to catch the one real error
      # That one error is "Jon owns students". The surface-alias table maps the
      # bare predicate "has" onto `owns`, so "my sister Maria" and "my mother"
      # become ownership claims about humans. Restricting the gate to
      # person-objects removes exactly those and keeps the rest.

    Scenario: Domains and ranges are multi-valued and satisfied by any match
      Given a property declaring the domains Organization, Person and Event
      When a Person subject is checked
      Then it conforms
      # Declared domain/range may be multi-valued; satisfying
      # one of them is conformance.
      # test: tests/ontology/test_cco_digest.py

    Scenario: Any-match also means an unmappable endpoint admits
      Given an object type that maps to no profile class
      When the range gate runs
      Then it admits, because there is nothing to judge it by

    Scenario: A blocked relation is annotated, not pruned
      Given a relation the domain or range gate refused
      When it is written
      Then the REL edge exists under its extracted predicate
      And ontology_blocked is true
      And ontology_property is null

    Scenario: The refusal outlives the chunk that made it
      Given chunk A asserts an edge with a subject it typed LOCATION
      And chunk B asserts the same edge with a subject it typed PERSON
      When both are written
      Then ontology_blocked stays true
      And the permissive write does not clear it
      # The gate runs per chunk and the same edge is written once per asserting
      # chunk; without a record of a REFUSAL, a refusal is indistinguishable from
      # having nothing to say (both arrive as NULL) and coalesce lets one
      # permissive write win.
      # test: tests/ontology/test_relation_conformance.py

  @ingest @ontology
  Rule: Conformance and hallucination are computed and logged per chunk

    Scenario: The rates are derived from the gate tallies
      Given 10 relations extracted from a chunk
      And 2 blocked on domain or range
      And 1 whose endpoint matches no extracted span
      When the chunk report is produced
      Then conformance is 0.8
      And hallucination is 0.1
      And the report also carries the endpoint-gate prune count
      # test: tests/ontology/test_relation_conformance.py

    Scenario: A chunk with no relations violated nothing
      When a chunk yields no relations
      Then conformance is 1.0 and hallucination is 0.0
      And no division by zero occurs

    Scenario: The resolved speaker counts as a supported endpoint
      Given a turn by "Gina" reading "My collection has twenty pieces"
      When the hallucination rate is computed
      Then the head "Gina" is treated as supported
      # Deixis resolution rewrites a first-person subject to the speaker's name,
      # which by design appears in no span of that chunk. Judging it against
      # spans alone made the rate read 1.000 on every conversational chunk
      # regardless of quality — a metric that measures nothing.

    Scenario: Hallucination is counted, never pruned on
      Given a relation whose endpoint matches no extracted span
      When the chunk is written
      Then the relation is still written if it clears the other gates
      And only the count changes
      # Pruning on this would be a retrieval-affecting change needing its own A/B.

    Scenario: The rates describe what the extractor produced
      Then they are counted over every extracted relation,
        including the ones the endpoint gate later drops

    Scenario: The report is logged at INFO only when a rate is imperfect
      Given a chunk with perfect conformance
      Then the line is logged at DEBUG
      Given a chunk with any refusal or unsupported endpoint
      Then the line is logged at INFO
      # The DEBUG line still distinguishes "gate ran, found nothing" from "gate
      # never ran". A gate nobody counts is indistinguishable from one that never
      # fires, which is how this layer sat at zero unnoticed.

  @ingest @frames @model
  Rule: Frames tag chunks, and frame instances fill roles

    Two distinct layers, both on by default.

    Scenario: A chunk is tagged with the frames its embedding evokes
      When a chunk is ingested with frames enabled
      Then its top 5 frames above a similarity of 0.20 become FRAME vertices
      And each is joined to the chunk by EVOKED_BY carrying the similarity
      And each FRAME node copies the frame's definition text and definition embedding
      And the chunk's already-computed embedding is reused, with no re-embed

    @measured
    Scenario: FRAME is canonical — one vertex per frame, not one per evocation
      Given a namespace in which 245 evocations name 58 distinct frames
      When the frames are written
      Then 58 FRAME vertices exist, MERGEd on the frame NAME
      And 245 EVOKED_BY edges exist, so no evidence was lost
      And FRAME carries no session_id and no chunk_id
      And everything per-evocation lives on the EVOKED_BY edge
      # It used to be one vertex per (session, chunk, rank), which left no
      # stable node to hang FRAME->FRAME edges off and forced the saturation
      # guard to count coverage off the node.
      # test: tests/storage/test_graph_store.py::test_write_frames_merges_one_canonical_vertex_per_frame_type

    Scenario: The per-frame data is written once, not once per chunk
      When the same frame is evoked by a second chunk
      Then its definition embedding is not re-sent
      And its BROADER parents are not re-written
      And emptying the database makes both eligible again
      # An embedding-less FRAME is invisible to the channel's vector search, so
      # a cache surviving a dropped namespace would kill the layer silently
      # rather than fail.
      # test: tests/storage/test_graph_store.py::test_emptying_the_database_reinstates_the_frame_writes

    Scenario: The dead FRAME columns are gone
      When the schema is ensured
      Then FRAME declares no lu, speaker, fes or fillers property
      # All four were declared and written EMPTY on every row by the only
      # writer. The role-filled frame data lives on FRAME_INSTANCE/PLAYS_ROLE,
      # which has a live reader.

    Scenario: An evoked frame brings its FrameNet parents as BROADER edges
      Given a chunk evoking "Revenge"
      When the frames are written
      Then a BROADER edge to "Rewards_and_punishments" exists
      And the parent is MERGEd bare: no summary, no embedding, no EVOKED_BY
      And a parent therefore never enters the ANN pool on its own account
      # Verified: 27 bare parents of 58 frames, none of which leaked into a
      # vector-search result. LSM_VECTOR skips null-embedding rows.

    Scenario: The frame taxonomy is strictly directional
      When the frame hierarchy is built
      Then Inheritance, Using and Perspective_on are walked sub -> super only
      And Subframe and Precedes are excluded
      And the result contains 0 symmetric pairs, against 254 in the query-side map
      # The query-side expansion map deliberately makes Perspective_on
      # symmetric, which is right for widening a search and corrupting for a
      # taxonomy: a symmetric `broader` makes the narrower-walk hand back
      # siblings as if they were specialisations. Subframe is part-of and
      # Precedes is temporal order; neither is subsumption, and nothing reads
      # them.

    Scenario: A frame instance is a role-filled reading of an evoked frame
      When frame-guided role extraction runs
      Then the chunk's candidate frames come from its main-verb lexical units, ranked by embedding
      And each candidate frame's CORE frame elements become a zero-shot extraction schema
      And filled roles become PLAYS_ROLE edges to entities
      And the instance is joined to its evidence chunk by FRAME_EVOKED_IN

    Scenario: The role-fillability gate keeps the layer quiet on filler
      Given the gate requires at least 2 filled core roles
      And at least one fill at or above a confidence of 0.55
      When conversational filler is processed
      Then no frame instance is produced
      # An instance under two filled roles is not a fact, it is a frame label
      # with a stray span attached.

    Scenario: Nested fills of one role collapse to the shortest
      Given the role "Goal" filled with "shopping experience",
        "special shopping experience" and "make a special shopping experience"
      When the fills are deduplicated
      Then only "shopping experience" survives
      # The shortest is the noun phrase; the longer ones picked up the verb and
      # the modifiers around it.

    Scenario: Role fillers pass the shared junk gate
      When a role filler is not a graph entity name
      Then it is dropped before any entity node exists

    @measured
    Scenario: A frame instance is identified by the fills that can persist
      Given two chunks each yielding "Activity_ongoing" with Agent Gina and Jon
      And one of them also fills a role with a span no entity was minted for
      When the instances are written
      Then both MERGE onto ONE node, because the extra fill cannot persist
      And that node carries a FRAME_EVOKED_IN edge per evidence chunk
      Given an instance whose every filler is unknown
      Then it is not written at all
      # A role edge MATCHes its entity and never mints one, so an unrecognised
      # filler is dropped — but it was hashed into the id anyway, which made
      # identity depend on something absent from the graph. Measured on conv-30
      # before the fix: 20 nodes carried 7 signatures, 13 of them duplicates
      # indistinguishable once written, each hoarding a single evidence edge
      # instead of pooling. After: 104 instances -> 92, duplicate signatures
      # 7 -> 1, instances with more than one evidence chunk 3 -> 8, and
      # FRAME_EVOKED_IN unchanged at 107 — nothing lost, only pooled.
      # test: tests/storage/test_graph_store.py::test_frame_instance_identity_ignores_fillers_that_cannot_persist

    Scenario: One residual duplicate survives, and it is an ordering artifact
      Given entity resolution runs at FLUSH, after frame instances are written
      When it merges or drops a filler entity that existed at write time
      Then two instances whose hashes differed legitimately can end up
        indistinguishable in the final graph
      # 1 node in 92 on conv-30. Repairing it means re-hashing the layer after
      # resolution, which is a whole post-flush pass for one node; recorded
      # rather than built.

    Scenario: The frame passes reuse the already-loaded models
      When frame instances are extracted
      Then the entity extractor's spaCy pipeline and extraction model are reused
      And no second copy of either is loaded
      And inference is serialised, because the model is shared across
        concurrently-ingesting conversations and is not reentrant

  @ingest @temporal @arcadedb
  Rule: Dates become canonical temporal nodes, not entities

    Scenario: A resolved date links the chunk to a global temporal node
      Given a chunk mentioning a date
      When it is ingested
      Then the date is resolved to ISO form against the chunk's anchor timestamp
      And a TEMPORAL node keyed on that ISO string is merged
      And the chunk is joined to it by MENTIONS_DATE

    Scenario: The anchor that resolves relative dates also dates the chunk
      Given a turn carrying a timestamp in its metadata
      When it is ingested
      Then CHUNK.ts is that timestamp
      And relative dates in the text resolve against the same anchor

  @ingest @arcadedb
  Rule: Writes are merges, so re-ingesting the same content is idempotent

    Scenario: A chunk is upserted on its identifier
      When the identical chunk is ingested twice
      Then one CHUNK record exists
      And its state and creation timestamp are preserved from the first write

    Scenario: An entity is merged on its normalised name
      When two chunks mention "Ada Lovelace"
      Then one ENTITY node exists
      And both chunks are joined to it by MENTIONS
      # Entities are namespace-global from the moment they are written. They are
      # NOT per-chunk, and there is no cross-chunk merge deferred to the flush.

    Scenario: A relation edge accumulates evidence rather than duplicating
      Given the same head, predicate and tail asserted by two chunks
      When both are written
      Then one REL edge exists
      And its evidence list holds both chunk identifiers
      And its confidence is the highest either assertion gave it

    @measured
    Scenario: Relation endpoints get provenance back to their chunk
      When a relation is written
      Then both endpoints are joined to the asserting chunk by MENTIONS
      And the chunk's own speaker is excluded from that write
      # Without the provenance, an entity that only ever appears as a relation
      # argument has no edge to any chunk: measured, 4 of 49 endpoints were
      # reachable from no chunk at all. Writing the SPEAKER too added 62 mention
      # edges to the two most saturated entities in the graph — precisely the
      # ones retrieval already has to de-weight — duplicating, in a weaker form,
      # what CHUNK.speaker already carries.

    Scenario: A single-valued predicate tombstones its predecessor
      Given the predicate "WORKS_AT" is in the single-valued set
      And an entity already has a current WORKS_AT edge elsewhere
      When a new WORKS_AT edge is written
      Then the old edge records invalid_at and is_current false
      And the old edge is NOT deleted
      And the new edge is the only current one
      # The single-valued set is a hardcoded list of 12 predicate labels. There
      # is no functional-property declaration read from the ontology.

    @measured
    Scenario: No predicate the pipeline currently emits can reach that path
      Given the single-valued set holds WORKS_AT, LIVES_IN, MARRIED_TO and 9 more
      And the shipped relation vocabulary is the profile's 25 camelCase properties
      When the two are compared, upper-cased, as the supersede check compares them
      Then the intersection is EMPTY
      And no relation written by the current extractor tombstones anything
      # WORKSFOR is not WORKS_AT, HOMELOCATION is not LIVES_IN. The mechanism is
      # implemented and correct; its trigger list predates the relation-vocabulary
      # rewrite and was never re-derived from it. This is exactly the shape this
      # repo keeps shipping: a gate that runs, passes and does nothing. Do not
      # "fix" it by string-matching — see the Section F item on reading
      # functionality from the ontology instead.

    Scenario: Ingest concurrency defaults to sequential
      Then the per-chunk concurrency cap is 1 unless opted into
      # The ENTITY name_norm index is UNIQUE, so concurrent merges of a shared
      # entity name across chunks race on the unique key.

    Scenario: One failed chunk does not abandon the document
      Given a document of 40 chunks, one of which raises
      When it is ingested
      Then the other 39 are written
      And the failure is recorded per chunk in the result's errors

  # ===========================================================================
  # SECTION D — CONSOLIDATION (THE FLUSH)
  # ===========================================================================

  @flush
  Rule: A flush drains the turn buffer, resolves entities, runs analytics, flips state

    The short-term buffer is genuinely cheap: a TURN vertex is written with no
    extraction and no embedding. The flush is where those turns meet the real
    pipeline, which is why draining is now the slowest step, not the analytics.

    Scenario: The flush runs its stages in order
      When a session is flushed
      Then the stages run in this order
        | stage                   | effect                                              |
        | list_turns              | read the session's raw TURN buffer                  |
        | drain_turns             | ingest each buffered turn through the real pipeline |
        | mark_turns_consolidated | flip the drained turns so they never drain twice    |
        | resolve_entities        | the layered entity resolver                         |
        | pagerank                | native PageRank onto ENTITY.pagerank                |
        | graph_embeddings        | node2vec onto ENTITY.graph_embedding                |
        | communities             | native Louvain onto ENTITY.community_id             |
        | topics                  | the topic stage over those communities              |
        | consolidate_session     | flip the session's chunks raw -> consolidated       |

    Scenario: The caller's per-turn extraction choice survives the buffer
      Given a turn buffered with extraction disabled
      When the buffer is drained
      Then that turn is re-ingested with extraction still disabled
      # Without carrying the flag, flush re-ingests every turn under one shared
      # setting and the caller's per-turn choice is silently lost at the boundary.

    Scenario: Each stage fails independently and is named in the result
      Given the community stage raises
      When the flush runs
      Then the error is recorded as "communities: <message>"
      And the later stages still run
      And consolidate_session still happens
      # An analytics failure must not skip the state flip, which is the thing
      # that makes the flush meaningful.

    Scenario: The analytical view is rebuilt once and shared
      When the flush reaches the analytics
      Then the OLAP analytical view is ensured once
      And PageRank, node2vec and Louvain each skip re-ensuring it
      And a failure to build the view is logged but not recorded as a stage error
      # Each algo still runs and soft-fails to 0 on its own; it just retries the
      # already-failed view.

    Scenario: The flush reports its counts
      When the flush completes
      Then it returns the turns ingested, the chunks they produced, the chunks
        flipped, the entities dropped, the entities merged, the graph embeddings
        computed, the communities computed, the topics created, the modularity
        and the elapsed time

  @flush
  Rule: Entity resolution is four ordered layers, most confident first

    Each layer re-reads the shrinking entity set, so a later layer sees the
    earlier layers' effects.

    Scenario: Layer 1 deletes generic non-referential hub entities
      Given entities named "others", "many", "kind" and "everyone"
      When resolution runs
      Then those nodes are detached and deleted

    Scenario: Layer 2 collapses plural variants of non-person entities
      Given the entities "path" and "paths"
      When resolution runs
      Then they merge into one
      And "Chris" and "James" are untouched, because persons are skipped here
      And proper nouns typed ORG, GPE, LOC and the rest are also skipped
      # A team "Warriors" must not collapse onto the common noun "warrior", nor
      # "Philippines" onto "philippine".

    Scenario: Gerunds are deliberately not stripped
      Then "building" does not become "build", nor "meeting" "meet"
      # Over-merging distinct concepts is worse than leaving a duplicate.

    @measured
    Scenario: The standardization layer elects the SINGULAR surface
      Given the group "style" and "styles"
      When the canonical member is chosen
      Then it is "style"
      # A longest-surface tiebreak elects the plural every time, so a layer whose
      # whole job is to collapse plural variants standardised onto the plural.
      # Measured: 38 of 2491 mention edges pointed at a name absent from their own
      # chunk text — `downs` x8, `dreams` x7 — because the surviving node was
      # named for a form that chunk never used.

    Scenario: Layer 3 collapses mentions sharing a syntactic head
      Given the entities "dance studio", "ideal dance studio" and
        "Jon's ideal dance studio by the water"
      When resolution runs
      Then all three merge onto "dance studio"
      And "Door Dash" is untouched, because its HEAD is a proper noun
      And "Jon's new store" reduces to "store", because a proper noun in
        MODIFIER position is a possessor whose claim belongs in a relation
      And the shortest surviving surface is elected
      And two mentions merge only when their TYPE also agrees
      # Layer 2 strips a determiner and singularizes, which cannot reach
      # variants differing by an adjective or a trailing prepositional phrase.
      # Measured: one dance studio was split across six surfaces, and the
      # duplicates produced self-referential edges that read as structure while
      # carrying nothing.
      # test: tests/ingestion/consolidation/test_syntactic_merge.py

    Scenario: Layer 4 merges a short person name into its person superstring
      Given "Mel" and "Melanie" are both person-like
      And "Mel" is a single alphabetic token of at least 3 characters
      And "Mel" is a strict case-insensitive prefix of "Melanie"
      When resolution runs
      Then "Mel" merges into "Melanie"
      And "Caroline" is untouched, because it is not a prefix
      And a non-person prefix is untouched, because both sides must be person-like

    Scenario: An ambiguous prefix is left split
      Given "ann" prefixes both "Anna" and "Anne" with close mention counts
      When resolution runs
      Then no merge happens
      # The top anchor must dominate the runner-up by a factor of 2, which lets
      # "mel" collapse into "Melanie" despite a stray "Mell" typo.

    Scenario: A malformed name may never anchor a nickname merge
      Given the junk name "caroline: thanks, melanie" exists
      When resolution runs
      Then "caroline" does not merge into it
      # "caroline" is a prefix of it, and merging the REAL person into the junk
      # name inverts the repair this layer performs.

    @measured
    Scenario: Embedding similarity is deliberately not the primary mechanism
      Given the true nickname pair Mel and Melanie at embedding cosine 0.528
      And the two distinct co-occurring people Melanie and Caroline at 0.497
      Then a similarity threshold cannot separate them
      And a conservative same-type co-occurring layer at 0.92 is a documented,
        off-by-default hook that is NOT run

    Scenario: A merge keeps every edge and its provenance
      When two entities merge
      Then the loser's relation edges are re-created on the winner through the
        normal relation write, keeping their evidence chunk lists
      And the loser's mention edges are re-pointed at the winner
      And the merge is idempotent and safe against merging a node into itself

    Scenario: Two repair layers were deleted rather than kept as insurance
      Then no speaker-prefix merge and no temporal/junk sweep run at flush time
      # Both existed only to repair graphs written before the mint-point gates
      # landed. Every write now passes the span reducer and the junk gate before
      # an ENTITY node exists at all.

  @flush @arcadedb @slow
  Rule: Graph analytics run natively in ArcadeDB, with known traps

    Scenario: The algorithms are only reachable through Cypher after a view rebuild
      When PageRank, node2vec or Louvain is invoked
      Then the call is issued with language "cypher", never SQL
      And the OLAP analytical view has been created and rebuilt through SQL first
      # Missing either yields an opaque 500.

    @measured
    Scenario: Louvain ignores its parameter map
      Given a resolution or granularity parameter passed to Louvain
      When community detection runs
      Then the parameter has no effect
      # Verified against the pinned build. There is no granularity knob.

    @measured
    Scenario: The database's own modularity score is not Newman modularity
      Given a partition placing every node in one cluster
      When the database's modularityScore is read
      Then it reports 0.750
      And Newman's Q for that same partition is 0.247
      Then the flush reports Newman modularity, computed in the service
      And the database's score is recorded only under a name marking it untrustworthy
      # Never tune on the native score.

    @measured
    Scenario: node2vec skips sparsely connected nodes
      Given entities with fewer than 3 edges
      When graph embeddings are computed
      Then those entities receive none
      # Measured coverage on one conversation: 3%. Check entity coverage before
      # trusting anything built on these vectors.

  @flush
  Rule: The topic layer defaults to producing nothing

    Scenario: Topics are off in the default mode
      Given the default mode
      When the topic mode is resolved
      Then it is "none"
      And the flush's topic stage creates no topic nodes

    Scenario: The topic stage is one implementation, invoked two ways
      Then the flush's topic stage and the standalone topic rebuild are the same code
      # A rebuild at new settings is never a different code path from the flush.

    Scenario: When topics do run they cluster the analytics output
      Given a topic source of "community"
      When the topic stage runs
      Then it reuses the Louvain partition
      Given a topic source of "graph_embedding"
      Then it clusters the node2vec vectors hierarchically
      And in both cases the reported modularity is Newman's Q over the relation edge list
      And each topic records its title, keywords, coherence and embedding

    Scenario: Topic membership is persisted on both sides
      When topics are persisted
      Then entities are joined to their topic by IN_TOPIC keyed on the normalised name
      And chunks are joined by HAS_TOPIC

  # ===========================================================================
  # SECTION E — RETRIEVAL
  # ===========================================================================

  @retrieve
  Rule: Four collectors always run, and optional channels join them

    The retriever spine is signal-blind: it merges what the collectors return,
    sorts by fused score and cuts to top_k. There is no query router, no intent
    classifier and no per-intent leg selection.

    @measured
    Scenario: No collector or channel reads REL edges
      When the channels that fired on a full conversation are censused
      Then they are semantic, text, frame, entity and temporal
      And no leg traverses a relation edge
      # This is why extraction quality does not move accuracy. Measured on
      # conv-30: an ingest producing 283 REL edges over 452 entities (40.3%
      # density, 31 distinct predicates) scored 0.791 against a baseline of
      # 0.807 with 55 edges over 719 entities (7.0% density, 13 predicates) —
      # flat, inside the +-0.02 judge-noise floor. Of 16 failures, 11 were
      # retrieval-bound. The relation layer is WRITTEN AND NEVER READ, which is
      # the severance this pipeline has carried since the graph-architecture
      # audit. Enriching extraction cannot fix it; a traversal leg would.

    Scenario: The baseline collectors are methods, not registered channels
      When retrieval runs
      Then the entity, vector, BM25 and temporal collectors all run, concurrently
      And they are not filtered by any profile or registry
      # They used to be Channel subclasses whose entire body was one delegating
      # call, behind a name-keyed registry and a profile filter that never
      # excluded anything.

    Scenario: A collector failure fails the request
      Given the vector collector raises
      When retrieval runs
      Then the error propagates
      # No return_exceptions swallow: a broken signal must surface immediately
      # rather than silently degrading recall.

    Scenario Outline: Optional channels are settings-gated
      Given the setting "<setting>"
      When the channel list is built
      Then the "<channel>" channel is "<state>"

      Examples: optional channels
        | setting                 | channel  | state          |
        | enable_frames           | frame    | on by default  |
        | enable_ontology_channel | ontology | OFF by default |

    @measured
    Scenario: The ontology channel is off because it has never been measured alone
      Given the ontology channel went live as a side effect of always-on injection
      When six channels were fused instead of five
      Then accuracy moved 0.766 -> 0.722 and retrieval misses 8 -> 13
      And that run also changed the extraction layer
      Then the channel is a suspect, not a convict, and is default-off so the
        next A/B varies one thing

    @measured
    Scenario: There is no graph-traversal channel
      Then no channel walks REL edges
      # A personalised-PageRank channel returned 367 hits for 12 gold-bearing
      # ones (3.3%, worst of six) and was the ONLY consumer of REL edges — which
      # is why every improvement to the relation layer measured flat on retrieval.

    Scenario: There is no topic channel
      Then nothing on the read path consumes what the topic stage writes
      # Its only reader was an orphaned retriever method. Note the reason is NOT
      # "topic_mode is none in both presets" — that claim is wrong; the
      # llm_assisted preset sets an LLM topic mode.

  @retrieve
  Rule: The pool is wider than the answer

    @measured
    Scenario: Each collector is asked for a pool, not for the caller's page
      Given a caller asking for 25 results
      When the collectors run
      Then each is asked for at least 100 candidates
      And evidence recall at 25 is 0.692 with a pool of 100, against 0.589 with a pool of 25
      # Fusion can only rank what it was handed. With pool == top_k, a chunk no
      # single collector ranked in its own top-25 could never be recovered by
      # agreement between them.

  @retrieve @model
  Rule: The query is embedded once, with a declarative variant

    Scenario: The interrogative prefix is stripped to make a second embedding
      Given the query "What is Caroline's profession?"
      When the query is analysed
      Then a declarative variant "Caroline's profession" is produced
      And both are embedded in one batched call
      And the variant is dropped when stripping changes nothing or leaves under 5 characters
      # The declarative form sits closer to how answers appear in chunks.

    Scenario: A chunk found by both embeddings contributes once
      Given a chunk surfaced by the query and by its variant
      When the vector collector emits candidates
      Then it emits one contribution, at the better of the two ranks
      # Otherwise a single logical source pays twice into the fusion.

    Scenario: Two token sets are derived from the query
      When the query is tokenised
      Then the graph token set keeps tokens of 3 characters or more
      And the lexical token set keeps only tokens longer than 3
      # Short tokens (month names, short proper nouns) matter where substring-of-
      # name is the lookup key, and are noise for the text scan.

  @retrieve @cypher
  Rule: Each collector has one job

    Scenario: The entity collector reaches chunks through mentions
      When the entity collector runs
      Then query tokens are matched against ENTITY.name_norm
      And tokens shorter than 5 characters match exactly, longer ones by CONTAINS
      And the chunks those entities are mentioned in are returned
      And the contribution carries a rank boost of 2, because entity hits are high-precision
      # Exact match on short tokens is what stops "war" matching "edward".

    Scenario: The vector collector is a dense ANN over chunk embeddings
      When the vector collector runs
      Then it searches CHUNK.embedding, filtered by session and state
      And each candidate's cosine is folded in as a small tiebreaker on top of its rank

    Scenario: The BM25 collector re-ranks a broad OR-pool
      When the BM25 collector runs
      Then an OR-pool three times the requested size is fetched by CONTAINS scan
      And it is re-ranked in memory with Okapi BM25
      And chunks are kept because they CONTAIN a query term, not because they score above zero
      And the BM25 score is clamped at 0 before it is folded in
      # Okapi IDF goes NEGATIVE once a term appears in more than about half the
      # corpus — routine for these small pools. A "score > 0" filter discards
      # genuine matches and could empty the channel; a negative addend would
      # demote a chunk below worse-ranked ones, inverting the tiebreaker.
      # test: tests/retrieval/test_channel_liveness.py

    Scenario: Missing rank-bm25 is a broken install, not a degraded mode
      Given rank_bm25 cannot be imported
      When the retriever module is imported
      Then a BrokenInstallError naming the extra is raised
      And there is no TF fallback
      # There was one, silently, for months: undeclared dependency -> uninstalled
      # -> silent tf scoring. It is gone. (The BM25 method's own docstring still
      # claims a fallback exists — the docstring is stale, the code is not.)

    @measured
    Scenario: The temporal collector has two shapes because one query shape has no date
      Given the query "what happened on 1 February 2023"
      When the temporal collector runs
      Then chunks linked to that TEMPORAL node are returned
      Given the query "when did Gina lose her job"
      When the temporal collector runs
      Then date-bearing chunks ABOUT the query's entities are returned instead
      # A date-keyed lookup only serves the first shape. Measured: 26 of 26
      # temporal questions stated no date, and the channel fired on 2 of 81
      # queries — both of which happened to quote a literal date.

    Scenario: The temporal fallback uses the same tokens as the entity collector
      Then both match on the graph token set
      # So the two channels agree on what the query is about.

  @retrieve @frames @ontology
  Rule: The optional channels are vector searches over persisted definitions

    Scenario: The frame channel matches the query against frames evoked here
      When the frame channel runs
      Then the query embedding is ANN-searched against FRAME.embedding in this namespace
      And each matched frame joins to the chunks that evoke it
      And each chunk scores by the frame's match TIMES its own EVOKED_BY confidence
      And chunks are scored by the sum of their frames' scores
      # The confidence weight is not decoration. A canonical FRAME fans out to
      # every chunk that evoked it, so without it all of them rank identically.

    Scenario: A matched frame also reaches its narrower frames
      Given a query matching "Rewards_and_punishments"
      When the frame channel runs
      Then chunks that only evoked "Revenge" are also collected
      And they score at 0.6 of a direct hit, so an exact match always wins
      And the expansion is one level, walked over the persisted BROADER edges
      # This is the READER that keeps the frame taxonomy from being another
      # write-only layer. Verified live across 6 namespaces: 33 chunks were
      # reachable ONLY through BROADER. The same helper serves the ontology
      # class hierarchy — only the vertex type and its key differ.

    @measured
    Scenario: A frame evoked by more than half a session's chunks is dropped
      Given a generic frame covering 68% of a session's chunks
      When the frame channel runs
      Then that frame's chunks are dropped from the join
      And coverage is measured over the GRAPH, never over the rows a query returned
      And coverage is scoped by the CHUNK's session, since FRAME has none
      # Counting from an ANN capped at top_k bounds the numerator by the sample
      # size while the denominator stays the whole session: the same frame
      # showed on 29% of rows while covering 68% of chunks, and so cleared the
      # 50% test on every query. Scoping on f.session_id after FRAME became
      # canonical would match nothing and silently retire the guard.

    Scenario: The ontology channel reaches chunks by two paths
      When the ontology channel runs
      Then a matched class reaches chunks directly through MATCHES_CLASS
      And indirectly through MENTIONS to an entity's INSTANCE_OF
      And the indirect path is discounted, since the entity may be incidental
      And a class matching more than half a session's chunks is dropped as saturated
      # The indirect path is much denser: on one conversation, 767 INSTANCE_OF
      # edges against 189 MATCHES_CLASS.

    Scenario: A matched class also reaches its subclasses
      Given a query matching "Facility"
      When the ontology channel runs
      Then chunks tagged "Dance Studio" are also collected
      And they score at 0.6 of a direct hit, so an exact match always wins
      And the expansion is one level, walked over the persisted BROADER edges
      And the walk matches on parent URI, which is unique, not on label
      # It used to pull every ONTOLOGY_CLASS row (1401 under CCO) into Python on
      # each query and re-split the `parents` string to rediscover the
      # hierarchy. CCO declares parents as LABELS and ONTOLOGY_CLASS(label) is
      # NOTUNIQUE, so the label is resolved to a URI at WRITE time — otherwise
      # the edge fans out across every class sharing that label.

    Scenario: An existing namespace is backfilled with the hierarchy
      Given a namespace whose ontology terms were persisted before BROADER existed
      When the ontology is written again
      Then the BROADER edges are added without re-MERGEing the class rows
      # write_ontology short-circuits on a term-count guard, and the subclass
      # expansion now reads ONLY these edges. Without the backfill such a
      # namespace keeps its classes and silently stops generalising.
      # test: tests/storage/test_graph_store.py::test_write_ontology_backfills_broader_on_an_existing_namespace
      # Generalisation is what the hierarchy is FOR: the chunk is about a
      # facility, it just never used the word. Without this the channel can only
      # return chunks whose tag the query matched head-on.

    @measured
    Scenario: The ontology channel remains OFF by default
      Then GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL defaults to 0
      # It has never been measured on its own. The subclass expansion above only
      # has an effect once it is switched on.

    Scenario: Channels contribute candidates rather than re-ranking
      Then a channel adds to the fusion and cannot displace a strong hit
      # Re-ranking the fused list on a graph signal has regressed retrieval here
      # before; see the community-detection experiment.

  @retrieve
  Rule: Candidates are fused by reciprocal rank and the fused order IS the ranking

    # ref: Cormack, Clarke & Buettcher, SIGIR 2009 — k = 60

    Scenario: Each candidate's contributions are summed
      Given a chunk returned by three collectors
      When the candidates are fused
      Then its score is the sum of one over 60 plus its rank in each, times that collector's boost
      And a chunk absent from a collector contributes nothing from it
      And the contributing source names are unioned onto the row

    Scenario: Fusion is unaffected by score scale
      Given the vector collector returns cosines and BM25 returns Okapi scores
      When the lists are fused
      Then the ranks drive the order and the raw scores act only as small tiebreakers

    Scenario: The sort by fused score decides what survives the cut
      When the fused list is ordered
      Then the pool is sorted by fused score descending
      And cut to the caller's top_k
      And each surviving row is given its final rank
      # This sort was once described as a mere tie-break ahead of a reranker.
      # It is the ranking.

    @measured
    Scenario: There is no cross-encoder reranker
      Then the module `graphknows.ranking.rerank` does not exist
      And the retriever holds no reranker attribute
      # It re-scored this same pool and measured WORSE at every depth a caller
      # uses — evidence recall at 25 of 0.567 against 0.692 without it — at 3.5x
      # the wall clock. A re-introduced reranker must come with an A/B, not a
      # deduction.
      # test: tests/retrieval/test_no_cross_encoder.py

    Scenario: Every candidate's stored row is fetched
      When the rows are shaped
      Then the stored chunk is fetched for every candidate, not only for some
      And each row carries chunk_id, doc_id, text, metadata, speaker, ts,
        rrf_score, the contributing source names, rank and entities
      # An old gate skipped candidates whose channel had set metadata, but the
      # fetch returns no metadata key at all — so the gate never saved the fetch
      # it claimed to and only starved those rows of speaker and ts, temporal
      # hits worst of all.

  @retrieve
  Rule: Neighbour expansion is off by default and merges rather than appends

    @measured
    Scenario: The default radius is zero
      Then no sibling chunks are merged unless a radius is configured
      # It HELPS long-document corpora (accuracy 0.371 -> 0.426 at radius 2) and
      # HURTS conversational memory, where short turns do not split and the
      # siblings are context-rot noise (0.895 -> 0.814 at radius 1).

    @measured
    Scenario: Siblings are stitched into the seed's own text
      Given a configured radius of 2
      When expansion runs
      Then each result's adjacent same-file chunks are merged INTO its text, in sequence order
      And the passage COUNT is unchanged
      And each sibling is claimed by only the first seed that takes it
      # Merging lifts evidence recall 0.305 -> 0.547 over the passages the
      # generator actually sees, where append-then-cut dropped it to 0.305 —
      # below the 0.353 no-expansion baseline — by spending the budget on the top
      # seeds' siblings rather than on distinct seeds.

    Scenario: An unavailable expansion is logged and skipped
      Given the neighbour lookup raises
      When expansion runs
      Then the un-expanded results are returned and the failure is logged

  @retrieve
  Rule: The result is plain dicts, with an optional structured fact-sheet

    Scenario: Every hit carries the chunk's own fields, from one builder
      When any collector contributes a candidate
      Then the payload is built by one shared helper, not hand-assembled
      And the fused row carries text, speaker, ts, entities, metadata and doc_id
      And a field the store already fetched therefore reaches the answerer
      # Each collector used to hand-build this dict and each kept a different
      # subset, so `ts` and `speaker` were dropped on five of six paths. They
      # survived only because _shape_and_rank re-fetches every candidate and
      # prefers the stored row — a backstop that made the omission invisible
      # rather than harmless.
      # test: tests/retrieval/test_det_retriever.py::test_a_hit_carries_the_chunks_timestamp

    @measured
    Scenario: A memory is dated from its ts, not by scraping its text
      Given a passage whose text contains no date and whose chunk has a ts
      When the memories are rendered for the answerer
      Then the block is prefixed with that date
      And passages are ordered chronologically by it
      Given a passage carrying an inline date and no ts
      Then the inline date still parses, so a flattened transcript is unaffected
      # THE most expensive defect measured in this repo. Scraping the text works
      # only when a date was flattened into it, which is true of a concatenated
      # transcript and false of a real conversation, where the timestamp is a
      # FIELD. Fed turn by turn, temporal accuracy was 0.077 against 0.962
      # document-fed; the generator said so itself, answering "the day before
      # the message (No specific date given) Hey Gina!" having retrieved exactly
      # the right chunk. Restoring the date moved accuracy 0.531 -> 0.864 and
      # temporal 0.077 -> 0.923 with evidence_recall BYTE-IDENTICAL — the whole
      # gap was generation-bound, not retrieval.
      # test: tests/evaluation/test_common.py::TestPassageTimestamp

    Scenario: There is no XML context payload
      When retrieval returns
      Then consumers read the fused result dicts directly
      And no XML document is assembled
      # A serialiser with graph_matches and semantic_matches sections used to
      # live here; both were permanently count="0" because the retriever never
      # populated them.

    Scenario: The fact-sheet surfaces the query entities' typed relations
      Given the fact context is enabled, which is the default
      When retrieval runs
      Then the query tokens are resolved to entity names
      And their typed relations and frame-instance role fills are returned as compact strings
      And the relation is named by its ontology property where one was recorded
      And only current, non-tombstoned edges are included

    Scenario: A fact-sheet failure never degrades recall
      Given the fact-sheet lookup raises
      When retrieval runs
      Then the passages are returned with an empty fact list
      And the failure is logged

    Scenario: The fact-context switch is read at call time
      Then flipping it per subprocess takes effect without re-import
      # So an A/B can vary it per server subprocess.

  @retrieve @cypher
  Rule: Caller-supplied Cypher is read-only and parameterised

    Scenario: String interpolation is refused
      Given a template containing a brace-delimited variable
      When it is executed
      Then a ValueError demands parameters instead

    Scenario: Write clauses are refused
      Given a template containing CREATE, MERGE, SET, DELETE, DETACH, REMOVE or DROP
      When it is executed
      Then a ValueError refuses it as not read-only

  # ===========================================================================
  # SECTION F — NOT IMPLEMENTED
  # ===========================================================================
  #
  # EVERYTHING BELOW THIS LINE IS INTENT, NOT BEHAVIOUR. None of it exists in
  # the code at HEAD. It is kept because each item represents a real design
  # decision worth revisiting — but a reader must never mistake it for a
  # description of the running system. Every Rule here is tagged
  # @not-implemented and every scenario is written in the future tense.
  #
  # Where something ADJACENT does exist, the Rule says so explicitly, because
  # "half of this is real" is the most dangerous shape a stale spec can take.
  # ===========================================================================

  @not-implemented
  Rule: FUTURE — Leiden community detection with the Constant Potts Model

    NOT IMPLEMENTED. Community detection is ArcadeDB's native Louvain, whose
    parameter map is ignored, and the default topic mode is "none" so the layer
    usually produces nothing at all. Leiden, CPM, resolution levels, partition
    records and per-community density/conductance statistics do not exist.
    # ref, if ever built: Traag, Waltman & van Eck 2019 (Sci. Rep. 9:5233) —
    #   Louvain can produce badly connected communities; Leiden cannot.
    #   Traag et al. 2011 (CPM); Fortunato & Barthelemy 2007 (resolution limit).

    Scenario: Communities will be detected with Leiden under CPM
      When community detection runs with default configuration
      Then the quality function will be the Constant Potts Model
      And every detected community will be internally connected
      And the run will record its quality function, resolution and seed

    Scenario: Hierarchical levels will be produced
      When community detection runs
      Then communities will be produced at each configured resolution level
      And each finer community will record its parent at the coarser level

    Scenario: Partition-level statistics will be stored once per partition
      Then modularity, CPM quality, community count, coverage, iterations, seed
        and graph revision will live on a partition record, not on a topic node

    Scenario: Per-community statistics will live on the topic node
      Then size, internal density, conductance and mean internal confidence
        will be recorded per topic
      And no topic will record the partition modularity as if it were its own

    Scenario: Detection will be reproducible given the same graph and seed

  @not-implemented
  Rule: FUTURE — Topic identifiers stable across flushes

    NOT IMPLEMENTED. Topic identifiers are minted per build. Nothing carries an
    identifier forward by membership overlap, and there is no record of a split
    or a merge.

    Scenario: A community whose membership barely changes will keep its identifier
      Given a topic whose newly detected community shares most of its members
      When topic identifiers are assigned
      Then the identifier will be carried forward and the membership updated

    Scenario: A split will produce one continuation and one new topic
      Then the larger community will keep the identifier
      And the smaller will record that it split from it

    Scenario: A merge will produce one continuation
      Then the absorbed topic will be recorded as merged, not deleted

  @not-implemented
  Rule: FUTURE — Bi-temporal querying "as of" a past time

    PARTIALLY BUILT, AND THE HALF THAT EXISTS IS DESCRIBED IN SECTION A. REL
    edges DO carry valid_at, invalid_at and is_current, and single-valued
    predicates ARE tombstoned rather than deleted. What does NOT exist: any
    ingest path that writes world-time (valid_at is only ever set by a caller
    that passes it, and no caller does), and any query that filters by a
    validity window — the function that did was deleted as unwired.

    Scenario: World-time will be extracted separately from system-time
      Given a chunk ingested today stating that Ada left the project in 2024
      When the fact is consolidated
      Then the assertion will record a world-time validity ending in 2024
      And a system-time of today

    Scenario: A query as of a past time will see what was believed then
      Given an assertion invalidated on 2026-06-01
      When memory is queried as of 2026-05-01
      Then the assertion will be returned as believed
      And a query as of 2026-07-01 will not return it

    Scenario: A temporal constraint in the query will restrict eligible facts
      Given the query "who did Ada work for in 2023"
      Then only facts believed valid in 2023 will be eligible

    Scenario: A contradiction that cannot be ordered will be flagged
      Given two assertions of a single-valued predicate with identical timestamps and confidence
      Then neither will be invalidated
      And retrieval will mark both as contested

    Scenario: Functionality will be read from the ontology
      Then the single-valued predicate set will come from the ontology's own
        declarations rather than from a hardcoded list of labels

  @not-implemented
  Rule: FUTURE — Privacy: detection, redaction and erasure

    NOT IMPLEMENTED. There is no PII detection, no redaction, no erasure path
    and no erasure certificate anywhere in the package. `purge_memory` and
    `drop_namespace` delete wholesale; they are not subject erasure.

    Scenario: Detected personal data will be labelled on the chunk
      Then the chunk will record the categories and the spans detected

    Scenario: Redaction will happen before embedding
      Then the embedding will be computed from the redacted text
      And no embedding derived from the unredacted text will be stored
      And no entity will be created from a span inside a redacted region

    Scenario: Erasing a chunk will remove everything derived from it
      Then the chunk, its solely-sourced relation edges, and the entities
        mentioned nowhere else will be deleted
      And entities mentioned elsewhere will keep their other mentions
      And the chunk's embedding will leave the vector index

    Scenario: Erasure of a data subject will produce a certificate
      Then every entity, edge and mention naming the subject will be removed
      And a certificate will record what was removed and when

    Scenario: Erasure will be idempotent
      Then erasing an already-erased chunk will succeed without change

  @not-implemented
  Rule: FUTURE — Security: provenance trust tiers and injection flagging

    NOT IMPLEMENTED. No trust tier is recorded, computed or carried anywhere.
    No text is examined for instruction-like content. Ingested text IS stored
    verbatim and never interpreted as configuration — but that is a property of
    the code having no such feature, not of a guard that was written.

    Scenario: Every chunk will carry a provenance trust tier
      Then a tier of verified, trusted or untrusted will be derived from the source
      And content claiming to be verified will not be able to raise its own tier

    Scenario: Instruction-like text will be flagged
      Then a chunk asking the reader to ignore previous instructions will be
        stored as data and flagged

    Scenario: The tier will reach the context payload
      Then an untrusted chunk will be emitted with its tier and enclosed so it
        is distinguishable from instructions
      And a fact asserted only by untrusted sources will be marked

  @not-implemented
  Rule: FUTURE — The XML context payload

    NOT IMPLEMENTED, AND DELETED ON PURPOSE. Consumers read the fused result
    dicts (Section E). The XML serialiser that existed had two sections
    permanently empty. Edge-weighted payload ordering, token budgeting, chunk
    collapsing and XML escaping all went with it.
    # ref, if ever rebuilt: Liu et al. 2024, "Lost in the Middle" (TACL
    #   12:157-173) — attention is weakest in the middle of a long context.

    Scenario: The payload will place the strongest evidence at the edges
      Then the highest-ranked chunk will appear first and the second-highest last
      And each chunk will still state its true rank

    Scenario: The payload will respect a token budget
      Then chunks will be added in rank order until the budget would be exceeded
      And the number omitted will be reported

    Scenario: Chunk text will be escaped so content cannot break the payload

    Scenario: Overlapping chunks will be collapsed with both provenances kept

  @not-implemented
  Rule: FUTURE — Per-leg latency budgets, abandonment and a retrieval trace

    NOT IMPLEMENTED. There are no per-collector budgets, no abandonment and no
    trace object: the trace builder was deleted with zero readers. A collector
    failure raises rather than degrading (Section E), which is deliberate — so
    "a failing leg does not fail the request" is NOT current behaviour either.

    Scenario: A leg exceeding its budget will be abandoned
      Then its partial results will be discarded rather than partially fused
      And the abandonment will be recorded in the trace

    Scenario: The trace will record each leg's contribution
      Then per leg: candidate count, latency, top ranks, and how many
        finally-selected chunks it contributed

    Scenario: A failing leg will be survivable
      Then the remaining legs will be fused and the response will name the failure

  @not-implemented
  Rule: FUTURE — Node embeddings by projection, and schema migration

    NOT IMPLEMENTED. node2vec runs, but inside ArcadeDB over the whole entity
    graph — there is no exported projection, no recorded parameter set, no seed
    guarantee, no graph revision on the vectors and no staleness marking. There
    is no migration framework at all: no revisions, no applied-migration record,
    no backfill, no confirmation gate on a destructive change, and no refusal of
    a mixed-model vector index.
    # ref, if the projection is ever built: Grover & Leskovec 2016, node2vec —
    #   p and q trade homophily against structural equivalence.

    Scenario: A typed projection will be exported before embedding
      Then the projection will be restricted to the configured node types
      And will record which types were included and the graph revision

    Scenario: The same graph and seed will produce the same embeddings

    Scenario: Embeddings will be keyed to the revision they describe
      Then an embedding older than the node's last structural change will be marked stale

    Scenario: A schema migration will be applied once and recorded
      Then applied revisions will be recorded in order with their timestamps
      And re-applying one will make no changes

    Scenario: Changing the embedding model will require a backfill
      Then retrieval will continue against the previous model until it completes
      And the switch will be atomic on completion
      And the backfill will resume from a checkpoint if interrupted

    Scenario: A mixed-model vector index will be refused
      Then a write from a second model of the same dimensionality will be refused,
        naming both model identifiers

  @not-implemented
  Rule: FUTURE — Ingestion durability: dedup, retries, dead-letter, back-pressure

    NOT IMPLEMENTED. There is no content hash, no simhash, no near-duplicate
    detection, no retry policy, no dead-letter queue and no back-pressure. What
    DOES exist and is described in Section C: merge-on-key writes, which make
    re-ingesting identical content idempotent, and a per-chunk failure that is
    recorded in the result's errors while the rest of the document proceeds.

    Scenario: A near-duplicate chunk will be linked rather than duplicated
      Then extraction will not be re-run over it

    Scenario: A failed extraction will still preserve the chunk
      Then the chunk will be marked pending extraction and queued for retry

    Scenario: A transient extraction failure will be retried with backoff

    Scenario: A chunk exhausting its retries will reach a dead-letter queue
      Then it will remain in memory marked as failed extraction

    Scenario: Ingestion will apply back-pressure rather than dropping work
      Then submission will block or be rejected with a retryable status

    Scenario: A flush will be idempotent under retry and resumable per stage
      Then re-requesting a completed flush will return its original result
      And a flush failing part-way will resume from its last successful stage
      And a second concurrent flush of one session will be refused

  @not-implemented
  Rule: FUTURE — Query routing by intent, and decomposition

    NOT IMPLEMENTED. Every collector runs on every query. There is no intent
    classifier, no per-intent leg selection, no multi-part decomposition and no
    global-summary path over topic summaries.
    # ref, if ever built: HippoRAG 2 (arXiv 2502.14802) — graph gains concentrate
    #   on associative tasks, so routing avoids paying for expansion that cannot
    #   help. Note that in THIS system the graph-traversal channel was measured
    #   and removed (Section E), so routing would have less to route.

    Scenario: Each intent will enable a different set of legs

    Scenario: A multi-part query will be decomposed into sub-queries
      Then each will be retrieved independently and their candidates fused

    Scenario: A global-summary query will be answered from topic summaries

    Scenario: Expansion will be constrained to the topics matching the query

  @not-implemented
  Rule: FUTURE — Extraction and retrieval quality gated in CI

    NOT IMPLEMENTED AS A GATE. The benchmark harnesses in `evaluation/` are real
    and are run by hand (LoCoMo, LongMemEval, BEAM). What does not exist: a
    recorded baseline per metric, an automatic regression comparison, a build
    that fails on a regression, or gold-set extraction metrics (triplet
    precision/recall/F1, frame identification accuracy, frame element F1).
    The judge's noise floor is plus or minus 0.02, so any such gate would need a
    tolerance wider than that before it could mean anything.

    Scenario: A regression beyond tolerance will fail the build
      Then the failing metric and its delta will be reported

    Scenario: Extraction metrics will be computed against a gold set
      Then triplet precision, recall and F1 will be reported
      And frame identification accuracy and frame element F1 will be reported

    Scenario: Stochastic stages will be seeded so two runs report the same scores

  @not-implemented
  Rule: FUTURE — Frame element filling as a parse, with null instantiation

    NOT IMPLEMENTED. Frame instances fill CORE frame elements with a zero-shot
    span extractor (Section C) and simply omit what does not fill. There is no
    trigger identification, no frame disambiguation against competing
    candidates, no coreness-set validation, no frame-to-frame relation traversal
    and no record of WHY an element is absent — the CNI/DNI/INI distinction is
    not represented anywhere.
    # ref, if ever built: Kalyanpur et al. 2020, Open-Domain Frame Semantic
    #   Parsing (arXiv 2010.10998); FrameNet null instantiation.

    Scenario: The evoking lexical unit will be identified as a trigger
      Then it will record its lemma, part of speech and character offsets

    Scenario: An omitted core element will record its null-instantiation kind
      Then it will be recorded as CNI, DNI or INI
      And no filler will be invented for it

    Scenario: A filler violating a coreness set will be rejected

    Scenario: The core element fill ratio will be reported per parse

  @not-implemented
  Rule: FUTURE — Graph metrics beyond PageRank

    NOT IMPLEMENTED. PageRank, node2vec vectors and a Louvain community id are
    persisted on ENTITY (Section D). Weighted degree, node specificity,
    clustering coefficient, core number, per-metric graph revisions, incremental
    recomputation over only the touched subgraph, and supernode flagging with a
    traversal fan-out cap do not exist.
    # ref, if ever built: node specificity as inverse passage frequency —
    #   HippoRAG (NeurIPS 2024).

    Scenario: Node specificity will act as a graph IDF
      Then a rarely-mentioned entity will score higher than a common one

    Scenario: Metrics will be recomputed only for the affected subgraph

    Scenario: A supernode will be flagged and its traversal capped

  @not-implemented
  Rule: FUTURE — Pruning the consolidated graph

    NOT IMPLEMENTED as a pruning stage. Entity resolution drops a fixed junk
    list and merges duplicates (Section D); nothing removes low-confidence
    edges, orphaned entities or duplicate assertions, and there is no reversible
    pruning log.

    Scenario: An entity with no edges and no mentions will be removed

    Scenario: An edge below the confidence floor will be removed and logged

    Scenario: Pruning will never remove a currently believed fact

    Scenario: Every pruning decision will be reversible within the retention window

  # ===========================================================================
  # OPEN QUESTIONS
  # Recorded so the spec states what is undecided rather than implying settled
  # behaviour. These are live questions about SHIPPED code — the deferred
  # features are Section F.
  # ===========================================================================
  #
  #  1. The ontology channel has never been measured on its own. It is
  #     default-off precisely so the next A/B can isolate it. Measure it, then
  #     record the number in `GraphKnowsSettings.enable_ontology_channel`.
  #  2. The relation label cap of 15 was fixed from a measured band (5 forces
  #     nonsense, ~14 produces sensible predicates, the model families are
  #     trained at <= 25 labels). The band's edges are not tight.
  #  3. The range gate's class-similarity floor of 0.7 sits at the midpoint of a
  #     gap only 0.26 wide, measured on ONE conversation. Re-census before
  #     retuning it.
  #  4. The relation verifier's end-to-end effect on answer accuracy has never
  #     been A/B'd. It is default-on on the strength of a per-edge probe.
  #  5. Newman modularity is a lying objective for this graph: the partition is
  #     computed over an unweighted relation load, and merging duplicate entity
  #     names LOWERS Q. It is reported, not optimised.
  #  6. Improvements to the relation layer have historically measured FLAT on
  #     retrieval, because the only channel that consumed REL edges was removed
  #     for being the worst of six. The fact-sheet is now the main consumer of
  #     that layer; its contribution is unmeasured.
  #  7. Judge noise is plus or minus 0.02. No single run may be read as evidence
  #     that a change helped.
