"""Symbolic knowledge: the symbol index, WordNet, imported ontologies, FrameNet.

This is *shared infrastructure*, not an optional signal: both ingestion and
retrieval import it. ``graphknows.channels`` holds the retrieval legs and may
import this package; this package must never import ``graphknows.channels``.

FrameNet is no longer a plane of its own here: ``framenet.emitter`` hands a
pack concepts and role predicates (FR-023), which are indexed by the embedding
of their definition exactly like an imported ontology's terms. There is no
frame-specific vertex or edge type, and therefore nothing to bridge between a
frame role and an ontology class.
"""
