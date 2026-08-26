"""Symbolic knowledge channels: FrameNet, WordNet, and imported ontologies.

Three resources that used to be two top-level packages reimplementing the same
six mechanisms — a definition-embedding index, a lemma+POS lexical index, a
taxonomy written to ``BROADER``, extractor hints, a graph writer, and a
saturation guard. They are one abstraction here, :class:`SymbolicChannel`.

This is *shared infrastructure*, not an optional signal: both ingestion and
retrieval import it. ``graphknows.channels`` holds the retrieval legs and may
import this package; this package must never import ``graphknows.channels``.
"""
