"""Golden-layer schema DDL for the single-database GraphStore.

Pure data: the ordered statement list ``ensure_schema`` executes. Kept out of
``graph_store.py`` so the store module reads as behaviour rather than 145 lines
of table definitions.
"""

from __future__ import annotations

import hashlib

from graphknows.storage.arcadedb._base import VEC_DIM_TOKEN

_VEC_DIM = VEC_DIM_TOKEN

# ---------------------------------------------------------------------------
# DDL — golden-layer schema (plan §1). Order matters: types before properties,
# properties before indexes. Executed idempotently in ensure_schema().
# ---------------------------------------------------------------------------

_CORE_DDL: list[tuple[str, str]] = [
    # ---- Vertex types -----------------------------------------------------
    # Singleton per database, recording the schema shape that created the graph
    # (see schema_version below). The UNIQUE index on stamp_id is what makes it a
    # singleton rather than a convention: a concurrent second insert is rejected
    # by the index, not by a race-prone read-then-write.
    ("sql", "CREATE VERTEX TYPE SCHEMA_STAMP IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SCHEMA_STAMP.stamp_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SCHEMA_STAMP.version IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SCHEMA_STAMP(stamp_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SESSION IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SESSION.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.started_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.consolidated_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSION(session_id) UNIQUE"),
    # Short-term buffer: one TURN per conversation turn, written with no
    # extraction and no embedding. flush() ingests them through the real
    # pipeline and flips state raw -> consolidated (see mark_turns_consolidated).
    ("sql", "CREATE VERTEX TYPE TURN IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TURN.turn_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.role IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.seq IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TURN.ts IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.state IF NOT EXISTS STRING"),
    # Who said it, when the caller knows a name rather than only a coarse role.
    ("sql", "CREATE PROPERTY TURN.speaker IF NOT EXISTS STRING"),
    # The caller's per-turn "extract from this?" decision, carried across the
    # buffer. Without it flush re-ingests every turn under one shared flag and
    # the caller's per-turn choice is silently lost at the flush boundary.
    ("sql", "CREATE PROPERTY TURN.infer IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE INDEX ON TURN(turn_id) UNIQUE"),
    ("sql", "CREATE INDEX ON TURN(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON TURN(state) NOTUNIQUE"),
    ("sql", "CREATE VERTEX TYPE FILE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FILE.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FILE(file_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE CHUNK IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CHUNK.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    # The retrieval surface: `text` verbatim, prefixed with the speaker when one
    # is known. `text` stays exactly what was said, so nothing that reads it
    # changes meaning; this is what gets embedded and lexically scanned, so a
    # query naming a person can reach an utterance that never names them.
    ("sql", "CREATE PROPERTY CHUNK.search_text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.seq IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.kind IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.ts IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.created_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.memory_class IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON CHUNK(chunk_id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(state) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # What serves the BM25 candidate pool (full_text_search_chunks). ArcadeDB
    # answers CONTAINSTEXT from a FULL_TEXT index and scans the bucket for
    # everything else — including the substring CONTAINS this replaced. BOTH
    # properties are indexed because the candidate query ORs across them, and
    # indexing only `text` would leave the `search_text` half a scan; dropping
    # that half instead would be a recall change hidden inside a plan change.
    ("sql", "CREATE INDEX IF NOT EXISTS ON CHUNK(text) FULL_TEXT"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON CHUNK(search_text) FULL_TEXT"),
    ("sql", "CREATE VERTEX TYPE ENTITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ENTITY.name_norm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.aliases IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY ENTITY.graph_embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON ENTITY(name_norm) UNIQUE"),
    ("sql", "CREATE INDEX ON ENTITY(state) NOTUNIQUE"),
    # Canonical temporal nodes — one global node per resolved ISO date string.
    ("sql", "CREATE VERTEX TYPE TEMPORAL IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TEMPORAL.date IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON TEMPORAL(date) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE TOPIC IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TOPIC.topic_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.modularity IF NOT EXISTS DOUBLE"),
    # Topic-stage outputs: a human-readable title, the distinctive terms that
    # separate this topic from the others, and how tight the cluster is.
    # No `embedding` and no vector index: the topic read path (a retrieval
    # channel that matched topics by vector, then traversed into their chunks)
    # was removed, so both were write-only. The centroid is still computed in
    # the builder — `coherence` is measured against it — just not stored.
    ("sql", "CREATE PROPERTY TOPIC.title IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.keywords IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.coherence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE INDEX ON TOPIC(topic_id) UNIQUE"),
    # Semantic frames (deterministic FrameNet layer). CANONICAL: one vertex per
    # frame type per namespace, keyed by the frame name. Everything per-evocation
    # lives on the EVOKED_BY edge to the CHUNK — there is no chunk_id and no
    # session_id here, because one frame is evoked by many chunks across many
    # sessions. That stable identity is what FRAME->FRAME BROADER edges hang off.
    #
    # `lu`, `speaker`, `fes` and `fillers` used to be declared here and were
    # written empty by the only writer on every single row. The role-filled frame
    # data lives on FRAME_INSTANCE/PLAYS_ROLE below, which has a live reader.
    ("sql", "CREATE VERTEX TYPE FRAME IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FRAME.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.frame_type IF NOT EXISTS STRING"),
    # The frame's definition text + its embedding, so a chunk's evoked frames are
    # vector-matchable by meaning against a query (see query_frames_by_embedding).
    ("sql", "CREATE PROPERTY FRAME.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON FRAME(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(frame_type) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON FRAME(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # FrameNet's frame-element (FE) plane -- coreness, Requires/Excludes,
    # CoreSets and semantic types (issue #143). Reference data, one vertex per
    # (frame, FE) pair keyed by "Frame:FE", written once per namespace by
    # GraphStore.write_frame_elements. Making coreness/CoreSet/Requires/
    # Excludes queryable and traversable here is separate from gating: the
    # validity stage in ingest reads them straight from
    # symbolic.framenet.fe, not through the graph.
    ("sql", "CREATE VERTEX TYPE FE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.frame IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.coreness IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.semtype IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FE.core_set IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON FE(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FE(coreness) NOTUNIQUE"),
    # FrameNet semantic types (fn.semtypes()) -- not vector-searched, no
    # embedding column, same precedent as ONTOLOGY_PROPERTY.
    ("sql", "CREATE VERTEX TYPE SEMTYPE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SEMTYPE.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEMTYPE.abbrev IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEMTYPE.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SEMTYPE(name) UNIQUE"),
    # Role-filled frame instances (WS/D3 layer). Distinct from the FRAME
    # chunk-tag layer above: one FRAME_INSTANCE = one evoked frame whose CORE
    # frame elements were GLiNER2-filled with entity fillers, linked to those
    # entities by PLAYS_ROLE {role} edges and to its evidence CHUNK by
    # FRAME_EVOKED_IN.
    ("sql", "CREATE VERTEX TYPE FRAME_INSTANCE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.frame IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.match_sim IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.role_confidence IF NOT EXISTS DOUBLE"),
    # How the instance is ASSERTED, from graphknows/linguistics.py. Without these
    # "she said she might move" is stored as "she moved". Every default is the
    # status quo, so a missed cue degrades to the old behaviour, never to a new
    # wrong claim; `modality_rules` records which rules fired, because a
    # heuristic that becomes a graph property has to be auditable.
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.polarity IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.modality IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.epistemic IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.hedged IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.voice IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.tense IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.attributed_to IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.modality_rules IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME_INSTANCE.lvc IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE INDEX ON FRAME_INSTANCE(id) UNIQUE"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON FRAME_INSTANCE(epistemic) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON FRAME_INSTANCE(frame) NOTUNIQUE"),
    # Ontology layer — the loaded ontology's classes, persisted per namespace so
    # they are joinable to the data they describe (an ENTITY links to its class
    # by INSTANCE_OF, a CHUNK to the classes it evokes by MATCHES_CLASS). Each
    # carries its natural-language definition and that definition's embedding, so
    # a query can vector-search the ontology and traverse into matching
    # chunks/entities. Present but empty when no ontology is configured — the DDL
    # is unconditional so an existing namespace upgrades on its next
    # ensure_schema.
    ("sql", "CREATE VERTEX TYPE ONTOLOGY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ONTOLOGY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY.source IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY.term_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON ONTOLOGY(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE ONTOLOGY_CLASS IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ONTOLOGY_CLASS.uri IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_CLASS.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_CLASS.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_CLASS.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON ONTOLOGY_CLASS(uri) UNIQUE"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON ONTOLOGY_CLASS(label) NOTUNIQUE"),
    (
        "sql",
        "CREATE INDEX IF NOT EXISTS ON ONTOLOGY_CLASS(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Ontology properties — reinstated (issue #140) with a reader:
    # GraphStore._functional_predicates queries `functional = true` rows to
    # drive relation-consolidation invalidation. Not vector-searched, so no
    # embedding column and no LSM_VECTOR index (only ONTOLOGY_CLASS is).
    ("sql", "CREATE VERTEX TYPE ONTOLOGY_PROPERTY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ONTOLOGY_PROPERTY.uri IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_PROPERTY.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_PROPERTY.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ONTOLOGY_PROPERTY.functional IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE INDEX ON ONTOLOGY_PROPERTY(uri) UNIQUE"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON ONTOLOGY_PROPERTY(label) NOTUNIQUE"),
    # One vertex per (channel, version) resource import (issue #145). Without
    # this a channel upgrade is invisible: nothing records which version a
    # namespace was populated from, so a corpus reimport silently keeps the
    # old plane forever and two versions of a grounding are indistinguishable.
    # `id` is the merge key ("<channel>@<version>") so two versions of one
    # channel are two vertices, not one overwritten row. Import counts are
    # supplied by the caller, which already computed them while writing.
    ("sql", "CREATE VERTEX TYPE SOURCE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SOURCE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.channel IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.source_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.imported_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SOURCE.imported IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SOURCE.dropped_expressions IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SOURCE.dropped_fe_mappings IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON SOURCE(id) UNIQUE"),
    ("sql", "CREATE INDEX ON SOURCE(channel) NOTUNIQUE"),
    # WordNet's synset plane (issue #142). Deliberately NOT eager: WordNet 3.0
    # has ~117k synsets and embedding all of them would put ~180MB into every
    # namespace for a corpus most namespaces never fully touch. Instead this
    # plane is minted LAZILY -- one SYNSET per grounding plus its shortest
    # hypernym chain (GraphStore.write_senses / write_synset_hypernyms) -- the
    # same precedent as SEMTYPE and ONTOLOGY_PROPERTY: no `embedding` property,
    # no LSM_VECTOR index, because nothing vector-searches it. PPR-based WSD
    # stays stubbed until this sense layer proves it has a reader.
    ("sql", "CREATE VERTEX TYPE SYNSET IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SYNSET.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SYNSET.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SYNSET.pos IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SYNSET.definition IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SYNSET(id) UNIQUE"),
    # `id` is `wn30:`-prefixed and built from the synset's numeric OFFSET+pos
    # (``synset_id()`` in graphknows/symbolic/wordnet.py), NOT its NAME
    # ("dog.n.01") -- the release prefix, not the name, is what is meant to
    # survive a corpus upgrade; `name` is carried separately for readability
    # and lookup only. `sense_key` is the stable cross-release bridge for the
    # sense itself (WordNet sense keys, unlike synset offsets, are guaranteed
    # stable across releases).
    ("sql", "CREATE VERTEX TYPE SENSE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SENSE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SENSE.sense_key IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SENSE.lemma IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SENSE.synset IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SENSE(id) UNIQUE"),
    ("sql", "CREATE INDEX ON SENSE(synset) NOTUNIQUE"),
    # ---- Edge types -------------------------------------------------------
    ("sql", "CREATE EDGE TYPE PART_OF IF NOT EXISTS"),  # CHUNK -> FILE
    ("sql", "CREATE EDGE TYPE IN_SESSION IF NOT EXISTS"),  # FILE|TURN -> SESSION
    ("sql", "CREATE EDGE TYPE MENTIONS IF NOT EXISTS"),  # CHUNK -> ENTITY
    ("sql", "CREATE PROPERTY MENTIONS.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE MENTIONS_DATE IF NOT EXISTS"),  # CHUNK -> TEMPORAL
    ("sql", "CREATE EDGE TYPE IN_TOPIC IF NOT EXISTS"),  # ENTITY -> TOPIC
    ("sql", "CREATE PROPERTY IN_TOPIC.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE EVOKED_BY IF NOT EXISTS"),  # FRAME -> CHUNK
    ("sql", "CREATE PROPERTY EVOKED_BY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE PLAYS_ROLE IF NOT EXISTS"),  # FRAME_INSTANCE -> ENTITY
    # FE plane edges (issue #143). No properties -- the relationship itself is
    # the fact; Requires/Excludes direction is exactly as FrameNet declares it.
    ("sql", "CREATE EDGE TYPE HAS_FE IF NOT EXISTS"),  # FRAME -> FE
    ("sql", "CREATE EDGE TYPE HAS_SEMTYPE IF NOT EXISTS"),  # FE -> SEMTYPE
    ("sql", "CREATE EDGE TYPE REQUIRES_FE IF NOT EXISTS"),  # FE -> FE
    ("sql", "CREATE EDGE TYPE EXCLUDES_FE IF NOT EXISTS"),  # FE -> FE
    # SKOS broader — the subsumption taxonomy, shared by both hierarchies:
    # FRAME -> FRAME (FrameNet Inheritance/Using/Perspective_on) and
    # ONTOLOGY_CLASS -> ONTOLOGY_CLASS (CCO subClassOf). One edge type because
    # both are read the same way: one hop DOWN from a query match, at a
    # discount, to reach the specific thing a general query asked about.
    ("sql", "CREATE EDGE TYPE BROADER IF NOT EXISTS"),
    # Every semantic ENTITY->ENTITY fact, regardless of predicate. The predicate is
    # the `predicate` PROPERTY, not the type name: the vocabulary is open and mostly
    # single-use, so a type-per-predicate minted hundreds of near-empty types (and a
    # CREATE EDGE TYPE DDL round-trip on the ingest hot path) while readers already
    # keyed off r.predicate — the type name was never authoritative, because
    # reserved-word predicates had to be escaped (find -> REL_FIND).
    # `predicate` carries two vocabularies: a frame-derived value is prefixed
    # `fn:` (e.g. `fn:Giving`), a relation-vocabulary value is the bare label
    # (`owns`); the bare frame name itself always lives in `REL.frame`.
    ("sql", "CREATE EDGE TYPE REL IF NOT EXISTS"),  # ENTITY -> ENTITY
    ("sql", "CREATE PROPERTY REL.predicate IF NOT EXISTS STRING"),
    # Where the edge came from. 'frame' marks co-role adjacency projected from a
    # FRAME_INSTANCE, which the fact-sheet excludes (its frame facts already
    # state the same content, with the role names intact).
    ("sql", "CREATE PROPERTY REL.source IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.head_role IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.tail_role IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON REL (predicate) NOTUNIQUE"),
    ("sql", "CREATE PROPERTY PLAYS_ROLE.role IF NOT EXISTS STRING"),
    ("sql", "CREATE EDGE TYPE FRAME_EVOKED_IN IF NOT EXISTS"),  # FRAME_INSTANCE -> CHUNK
    # FRAME_INSTANCE -> FRAME. The fact plane's link into the frame taxonomy:
    # BROADER hangs off the canonical FRAME vertex, so without this edge no
    # instance can generalise through it.
    ("sql", "CREATE EDGE TYPE EVOKES IF NOT EXISTS"),
    # Frame nesting: a role span that embeds another frame's trigger ("...because
    # my laptop died" fills Explanation with a Causation whose Cause is a Death).
    # FRAME_INSTANCE -> FRAME_INSTANCE, the embedding role on the edge. Written
    # only on the SRL identification path, which is what yields FE spans; read by
    # entity_fact_context, so it is not another write-only layer.
    ("sql", "CREATE EDGE TYPE FRAME_NESTS IF NOT EXISTS"),  # FRAME_INSTANCE -> FRAME_INSTANCE
    ("sql", "CREATE PROPERTY FRAME_NESTS.role IF NOT EXISTS STRING"),
    # Ontology edges. INSTANCE_OF/MATCHES_CLASS carry the mapping score so a
    # weak match stays distinguishable from a confident one downstream.
    ("sql", "CREATE EDGE TYPE INSTANCE_OF IF NOT EXISTS"),  # ENTITY -> ONTOLOGY_CLASS
    ("sql", "CREATE PROPERTY INSTANCE_OF.score IF NOT EXISTS DOUBLE"),
    # Grounding provenance, written by write_groundings so a grounding edge is
    # attributable to the chunk, resource version and matcher that produced it.
    ("sql", "CREATE PROPERTY INSTANCE_OF.channel IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.matcher IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.rank IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY INSTANCE_OF.span IF NOT EXISTS STRING"),
    # Set true by supersede_groundings when a rebind promotes a newer version;
    # absent (not false) means "not superseded" for every edge written before
    # this property existed, so an old namespace's groundings are not
    # retroactively flagged.
    ("sql", "CREATE PROPERTY INSTANCE_OF.superseded IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE EDGE TYPE MATCHES_CLASS IF NOT EXISTS"),  # CHUNK -> ONTOLOGY_CLASS
    ("sql", "CREATE PROPERTY MATCHES_CLASS.sim IF NOT EXISTS DOUBLE"),
    # The ontology property a relation's predicate mapped onto, read back by
    # entity_fact_context to name the relation in the domain's own vocabulary.
    ("sql", "CREATE PROPERTY REL.ontology_property IF NOT EXISTS STRING"),
    # Sticky "do not annotate this edge". The ontology range gate runs per chunk,
    # and the same edge is written once per chunk that asserts it — so without a
    # record of a REFUSAL, a refusal is indistinguishable from having nothing to
    # say (both arrive as NULL) and coalesce lets a single permissive write win.
    ("sql", "CREATE PROPERTY REL.ontology_blocked IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY REL.verifier_score IF NOT EXISTS DOUBLE"),
    # Written by write_relation since it was added and never declared, so every
    # one was untyped and unindexed. Bi-temporality was half-built: the values
    # were there, nothing could query them.
    ("sql", "CREATE PROPERTY REL.fact IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY REL.valid_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.invalid_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.is_current IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY REL.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.created_at IF NOT EXISTS STRING"),
    # `canonical_predicate` is the STABLE identifier and the dedup key; `predicate`
    # stays the most SPECIFIC label, for display and for querying in the user's
    # own vocabulary. One field cannot do both: dedup wants stability, display
    # wants specificity, and with a single field a relex label that varies per
    # chunk (hasSpouse here, isMarriedTo there) splits one fact into two rows.
    ("sql", "CREATE PROPERTY REL.canonical_predicate IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.frame IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.frame_instance_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.reified IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE PROPERTY REL.anchor IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.polarity IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.modality IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.epistemic IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY REL.hedged IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON REL(canonical_predicate) NOTUNIQUE"),
    ("sql", "CREATE INDEX IF NOT EXISTS ON REL(is_current) NOTUNIQUE"),
    # How many distinct assertions (relex label variants of the same
    # canonical_predicate) collapsed onto this row — raised by
    # GraphStore.consolidate_relations, initialised to 1 by write_relation so
    # every edge has a count even before consolidation ever runs.
    ("sql", "CREATE PROPERTY REL.support IF NOT EXISTS INTEGER"),
    # ENTITY -> ENTITY, not REL -> REL: ArcadeDB edges connect vertices only,
    # so a contradicting PAIR of REL rows is identified by (head,
    # canonical_predicate) plus `current_tail` — the surviving edge's tail
    # name_norm — which together are enough to find both REL rows again.
    ("sql", "CREATE EDGE TYPE CONTRADICTS IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CONTRADICTS.canonical_predicate IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CONTRADICTS.reason IF NOT EXISTS STRING"),  # 'polarity' | 'functional'
    ("sql", "CREATE PROPERTY CONTRADICTS.current_tail IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CONTRADICTS.created_at IF NOT EXISTS STRING"),
    # BROADER stays ONE edge type across every taxonomy channel that feeds it
    # (FrameNet Inheritance/Using/Perspective_on, CCO subClassOf, and now
    # WordNet hypernymy), discriminated by `rel`. `transitive` is what a
    # closure query must check before following an edge: WordNet's
    # instance_hypernym relation (an individual's link to its class, e.g.
    # Mount Everest -> mountain) is written with `rel = 'instance_hypernym'`
    # and `transitive = false` so it is never chained past its one hop, unlike
    # ordinary `hypernym` edges.
    ("sql", "CREATE PROPERTY BROADER.rel IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY BROADER.transitive IF NOT EXISTS BOOLEAN"),
    ("sql", "CREATE EDGE TYPE IN_SYNSET IF NOT EXISTS"),  # SENSE -> SYNSET
    # Sense-level relations (antonym, derivationally_related_form, pertainym,
    # participle) live ONLY here, never on BROADER: the writer
    # (write_sense_relations) MATCHes `:SENSE` on both endpoints, so a
    # sense-level relation landing on a synset node is structurally
    # impossible rather than merely undocumented.
    ("sql", "CREATE EDGE TYPE SENSE_REL IF NOT EXISTS"),  # SENSE -> SENSE
    ("sql", "CREATE PROPERTY SENSE_REL.rel IF NOT EXISTS STRING"),
    # One WordNet grounding, in full provenance. The merge key is (entity,
    # chunk_id, span, synset) -- token-level, not (entity, synset) -- so the
    # same entity legitimately holds DIFFERENT senses in different chunks
    # instead of the second grounding overwriting the first.
    ("sql", "CREATE EDGE TYPE HAS_SENSE IF NOT EXISTS"),  # ENTITY -> SENSE
    ("sql", "CREATE PROPERTY HAS_SENSE.channel IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY HAS_SENSE.version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY HAS_SENSE.matcher IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY HAS_SENSE.score IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY HAS_SENSE.rank IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY HAS_SENSE.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY HAS_SENSE.span IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY HAS_SENSE.synset IF NOT EXISTS STRING"),
]


def schema_version(dims: int) -> str:
    """Short digest of the schema shape a graph was built with.

    Derived from the DDL statement text — ``_CORE_DDL`` is
    ``list[tuple[str, str]]``, so the statement is projected out of each pair —
    combined with ``dims``, which the DDL interpolates into the ``LSM_VECTOR``
    indexes: a graph built at one embedding dimension cannot serve another.

    Because the digest is over the text, editing ``_CORE_DDL`` at all — a
    reworded comment inside a statement included — is now a client-visible
    event: every existing graph is stranded and must be rebuilt from its
    original sources. There is no migration.
    """
    ddl_text = "\n".join(stmt for _lang, stmt in _CORE_DDL)
    return hashlib.sha256(f"{ddl_text}\n{dims}".encode()).hexdigest()[:16]
