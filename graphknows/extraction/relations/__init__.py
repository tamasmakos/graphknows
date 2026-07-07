"""Relation extraction strategies.

Two interchangeable strategies feed the relation graph:

- ``spacy`` — dependency-pattern extraction (LLM-free, deterministic). Always
  active in ``llm_free`` mode; off in ``llm_assisted`` mode (see
  ``graphknows.config.profiles.get_profile``).
- ``dspy`` — RDF-ontology-constrained LLM extraction (LLM-assisted). Wired through
  the ``IRelationExtractor`` port (``graphknows.ports``) and only
  active when the memory mode is ``llm_assisted``.

The GLiNER entity pass also emits candidate relations. Which strategies run is
decided by ``GraphKnowsSettings.mode`` via ``get_profile(settings)``
(see ``graphknows.config.profiles``).
"""
