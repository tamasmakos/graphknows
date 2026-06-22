"""Relation extraction strategies.

Two interchangeable strategies feed the relation graph:

- ``spacy`` — dependency-pattern extraction (LLM-free, deterministic). Toggle with
  ``GraphKnowsSettings.spacy_relations``.
- ``dspy`` — RDF-ontology-constrained LLM extraction (LLM-assisted). Wired through
  the ``IRelationExtractor`` port (``graphknows.config.adapters.ports``) and only
  active when the memory mode is ``llm_assisted``.

The GLiNER entity pass also emits candidate relations. Which strategies run is
decided by ``GraphKnowsSettings.mode`` (see ``graphknows.config``).
"""
