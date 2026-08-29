"""Symbolic knowledge channels: FrameNet, WordNet, and imported ontologies.

Three resources that used to be two top-level packages reimplementing the same
six mechanisms — a definition-embedding index, a lemma+POS lexical index, a
taxonomy written to ``BROADER``, extractor hints, a graph writer, and a
saturation guard. They are one abstraction here, :class:`SymbolicChannel`.

This is *shared infrastructure*, not an optional signal: both ingestion and
retrieval import it. ``graphknows.channels`` holds the retrieval legs and may
import this package; this package must never import ``graphknows.channels``.

Sharing those six mechanisms is not the same as sharing meaning. FrameNet
(``FRAME`` / ``FE`` / ``SEMTYPE``) and an imported ontology (``ONTOLOGY_CLASS`` /
``ONTOLOGY_PROPERTY``) are independent signals: no edge joins the two planes
directly. They meet by both attaching to the same ``CHUNK`` (a frame evoked
on a chunk, an ontology class matched on the same chunk), and, indirectly,
via ``FRAME_INSTANCE-[:PLAYS_ROLE]->ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`` —
but neither is a semantic mapping between a frame role and an ontology class.
A frame role filler is never ontologically typed by virtue of the role it
fills — only an explicit
``ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`` edge types an entity. A confidence-bearing
``ALIGNED_WITH`` bridge between a ``SEMTYPE``/``FE`` and an ``ONTOLOGY_CLASS`` has been
considered and deliberately not built: FrameNet ships no mapping from its
semantic types to any ontology's IRIs, so the edge would be a hand-curated
per-ontology guess, and nothing on the retrieval path would traverse it.
"""
